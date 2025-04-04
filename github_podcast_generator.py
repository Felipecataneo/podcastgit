import requests
import re
import os
import base64
import textwrap
from urllib.parse import urlparse
import time
from pathlib import Path
import json
from tqdm import tqdm
import io
import google.generativeai as genai
from gtts import gTTS
from dotenv import load_dotenv
from pydub import AudioSegment
import math # Importado para usar log

# Carrega variáveis de ambiente do .env (se existir)
load_dotenv()

# --- NOVAS CONSTANTES PARA AVALIAÇÃO DE COMPLEXIDADE ---
# Limiares para classificar a complexidade. Ajuste conforme necessário.
COMPLEXITY_THRESHOLD_CHARS_LOW = 50000      # Abaixo disso, provavelmente simples
COMPLEXITY_THRESHOLD_CHARS_HIGH = 700000    # Acima disso, provavelmente complexo
COMPLEXITY_THRESHOLD_FILES_LOW = 50         # Menos que isso, provavelmente simples
COMPLEXITY_THRESHOLD_FILES_HIGH = 600       # Mais que isso, provavelmente complexo
COMPLEXITY_THRESHOLD_LANGS_HIGH = 5         # Mais que isso, contribui para complexidade
# ---------------------------------------------------------

class GitHubPodcastGenerator:
    """
    Ferramenta para gerar podcasts didáticos em áudio explicando repositórios do GitHub,
    focando em desenvolvedores juniores, usando a API Gemini e gTTS.
    Versão adaptada para um modelo hipotético de contexto longo (ex: 1M tokens).
    Adapta o nível de detalhe do podcast com base na complexidade estimada do repositório.
    """

    def __init__(self, gemini_api_key=None, github_token=None):
        """
        Inicializa o gerador de podcast.

        Args:
            gemini_api_key (str, optional): Sua chave da API Gemini. Tenta obter do ambiente se não fornecida.
            github_token (str, optional): Seu token do GitHub. Tenta obter do ambiente se não fornecido.
        """
        self.gemini_api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY")
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")

        if not self.gemini_api_key:
            print("⚠️ Nenhuma chave da API Gemini encontrada. A geração do script falhará.")
            # Lançar um erro pode ser melhor em produção, mas deixamos seguir para a UI mostrar
            # raise ValueError("API Key da Gemini não encontrada.")
        else:
            try:
                genai.configure(api_key=self.gemini_api_key)
                print("✓ API Gemini configurada.")
            except Exception as e:
                print(f"Erro ao configurar a API Gemini: {e}")
                self.gemini_api_key = None # Invalida se falhar

        # --- ALTERAÇÃO: Nome do modelo hipotético ---
        #self.gemini_model_name = 'gemini-1.5-pro-preview-0409' # Exemplo de modelo real
        self.gemini_model_name = 'gemini-2.5-pro-exp-03-25' # Modelo mais rápido e recente (Preview)
        #self.gemini_model_name = 'gemini-1.5-pro-latest' # Usar o mais recente pro
        self._log_message(f"Usando modelo Gemini: {self.gemini_model_name}", "info")
        # ----------------------------------------------

        self.headers = {}
        if self.github_token:
            self.headers = {"Authorization": f"token {self.github_token}"}
            print("✓ Token do GitHub encontrado.")
        else:
            print("⚠️ Token do GitHub não encontrado. Repositórios privados ou com alto tráfego podem falhar.")

        # Detalhes do repositório
        self.repo_owner = None
        self.repo_name = None
        self.branch = "main"
        self.repo_url = None

        # Extensões de código a analisar
        self.code_extensions = ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.h', '.go', '.rb', '.php', '.ts', '.jsx', '.tsx', '.ipynb', '.sh', '.md', '.yaml', '.yml', '.json', '.xml', '.sql', '.dockerfile'] # Adicionado alguns mais comuns

        # Armazena dados do repositório
        self.code_files = []
        self.readme_content = ""
        self.repo_summary = {}
        self.complexity_level = 'medium' # Nível de complexidade padrão
        self.total_code_chars_collected = 0 # Armazena o total de caracteres de código lidos

        self.generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            # max_output_tokens=8192, # Pode precisar aumentar para respostas mais longas
            temperature=0.6 # Um pouco menos de criatividade para análises mais técnicas
        )

        # Configurações de segurança (manter ou ajustar conforme necessário)
        self.safety_settings={
            genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
        }

        # --- PARÂMETROS DE LIMITE (Ajustados) ---
        # Limite alto, mas evita repositórios GIGANTES travarem tudo só na leitura
        self.max_files_to_read_content = 1500 # Aumentado um pouco
        # Limite alto por arquivo (ex: ~150k caracteres)
        self.max_chars_per_file_read = 150000 # Aumentado
        # Limite para o README enviado à IA (ex: ~300k caracteres)
        self.max_readme_chars_for_ai = 300000 # Aumentado
        # Limite TOTAL de caracteres de CÓDIGO a serem enviados à IA (ex: 5 Milhões para Flash, pode ir mais pro Pro 1.5)
        # Deixa espaço para prompt, readme, lista de arquivos, resposta, etc.
        self.max_total_code_chars_for_ai = 5000000 # AUMENTADO SIGNIFICATIVAMENTE
        # Limite para a lista de nomes de arquivos enviada à IA
        self.max_file_list_for_ai = 3000 # Aumentado
        # -------------------------------------------------------------------


    def _log_message(self, message, level="info"):
        """Helper para logar mensagens."""
        prefix = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "success": "✅"}.get(level, "➡️")
        print(f"{prefix} {message}")

    def parse_github_url(self, url):
        """Analisa a URL do GitHub para extrair proprietário, nome do repo e branch opcional."""
        try:
            parsed_url = urlparse(url)
            if "github.com" not in parsed_url.netloc:
                raise ValueError("Não é uma URL válida do GitHub.")
            path_parts = parsed_url.path.strip('/').split('/')
            if len(path_parts) < 2:
                raise ValueError("URL não contém um caminho de repositório válido.")

            self.repo_owner = path_parts[0]
            self.repo_name = path_parts[1].replace('.git', '')
            self.repo_url = f"https://github.com/{self.repo_owner}/{self.repo_name}"

            # Resetar estado interno para nova análise
            self._reset_state()

            if len(path_parts) > 3 and path_parts[2] == "tree":
                self.branch = path_parts[3]
            else:
                self.branch = self._get_default_branch() # Tenta obter o default

            self._log_message(f"Repositório: {self.repo_owner}/{self.repo_name} (Branch: {self.branch})", "info")
            return True
        except ValueError as e:
            self._log_message(f"Erro ao analisar URL: {e}", "error")
            return False
        except Exception as e:
            self._log_message(f"Erro inesperado ao analisar URL: {e}", "error")
            return False

    def _reset_state(self):
        """Reseta o estado interno para permitir análises de múltiplos repositórios."""
        self.branch = "main" # Reseta para default inicial
        self.code_files = []
        self.readme_content = ""
        self.repo_summary = {}
        self.complexity_level = 'medium'
        self.total_code_chars_collected = 0

    def _get_default_branch(self):
        """Obtém a branch padrão do repositório via API do GitHub."""
        api_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}"
        try:
            response = requests.get(api_url, headers=self.headers)
            # Espera um pouco antes de potencialmente dar erro
            time.sleep(0.2)
            response.raise_for_status()
            repo_info = response.json()
            default_branch = repo_info.get("default_branch", "main")
            self._log_message(f"Branch padrão detectada: {default_branch}", "info")
            return default_branch
        except requests.RequestException as e:
            status_code = e.response.status_code if hasattr(e, 'response') and e.response is not None else 'N/A'
            if status_code == 404:
                 self._log_message(f"Repositório '{self.repo_owner}/{self.repo_name}' não encontrado ou privado. Usando 'main'.", "warning")
            elif status_code == 403:
                 self._log_message("Limite de taxa GitHub ou token inválido ao buscar branch. Usando 'main'.", "warning")
            else:
                 self._log_message(f"Não foi possível obter a branch padrão (usando 'main'): {e}", "warning")
            return "main" # Retorna 'main' como fallback seguro

    def fetch_repo_structure(self, progress_callback=None):
        """Busca a estrutura do repositório e arquivos importantes, com limites aumentados."""
        self._log_message("Buscando estrutura do repositório (modo contexto amplo)...", "info")
        self.code_files = []
        self.readme_content = ""
        self.total_code_chars_collected = 0 # Reseta contador de caracteres

        api_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/git/trees/{self.branch}?recursive=1"

        try:
            response = requests.get(api_url, headers=self.headers)
            response.raise_for_status()
            tree_data = response.json()

            if tree_data.get("truncated"):
                self._log_message("A árvore do repositório é muito grande e foi truncada pela API do GitHub. Analisando arquivos disponíveis.", "warning")

            all_items = tree_data.get("tree", [])
            total_items = len(all_items)
            processed_items = 0
            files_content_read_count = 0 # Contador para o novo limite

            # Filtrar apenas blobs (arquivos) para a barra de progresso de leitura
            blob_items = [item for item in all_items if item["type"] == "blob"]
            total_blobs = len(blob_items)
            blobs_processed_for_content = 0

            items_iterable = tqdm(blob_items, desc="Lendo conteúdo dos arquivos", unit="arquivo", disable=progress_callback is not None)

            for item in items_iterable:
                blobs_processed_for_content += 1
                processed_items += 1 # Conta como item processado geral
                if progress_callback:
                    # Progresso baseado na leitura de blobs, mas mostrando o path
                    progress_val = (blobs_processed_for_content / total_blobs) if total_blobs > 0 else 0
                    progress_callback(progress_val * 0.5, f"Lendo: {item['path']}") # 0% a 50% para leitura

                file_path = item["path"]
                file_name = os.path.basename(file_path)
                _, ext = os.path.splitext(file_name)
                ext_lower = ext.lower()

                # Processar README
                if file_name.lower() == "readme.md":
                    content = self._fetch_file_content_by_sha(item["sha"], file_path)
                    if content:
                        self.readme_content = content # Limite de chars aplicado na leitura
                        self._log_message(f"README.md encontrado e processado ({len(content)} chars).", "info")

                # Processar arquivos de código/texto relevantes
                elif ext_lower in self.code_extensions:
                    file_info = {
                        "path": file_path,
                        "name": file_name,
                        "extension": ext_lower,
                        "sha": item["sha"],
                        "content": None # Inicializa sem conteúdo
                    }
                    # --- Lógica de leitura de conteúdo com limite maior ---
                    if files_content_read_count < self.max_files_to_read_content:
                        # Adiciona uma pausa menor para modelos rápidos como Flash
                        time.sleep(0.05) # Reduzido para acelerar, ajustar se der rate limit
                        content = self._fetch_file_content_by_sha(item["sha"], file_path)
                        if content:
                            file_info["content"] = content # Limite de chars por arquivo aplicado na leitura
                            self.total_code_chars_collected += len(content) # Acumula caracteres LIDOS
                            files_content_read_count += 1
                            if files_content_read_count % 100 == 0: # Log a cada 100 arquivos lidos
                                self._log_message(f"Lido conteúdo de {files_content_read_count} arquivos... ({self.total_code_chars_collected / 1000:.1f}k chars)", "info")

                    # Mesmo se não ler o conteúdo, adiciona à lista para ter o path
                    self.code_files.append(file_info)
                # Ignorar outros tipos de arquivo para simplificar

            # --- Log do limite ---
            if files_content_read_count >= self.max_files_to_read_content:
                 self._log_message(f"Limite de {self.max_files_to_read_content} arquivos com conteúdo lido atingido.", "warning")
            self._log_message(f"Encontrados {len(self.code_files)} arquivos relevantes. Conteúdo lido para {files_content_read_count} arquivos (Total: {self.total_code_chars_collected / 1000:.1f}k caracteres).", "info")
            # -------------------------------

            if not self.readme_content:
                self._log_message("README.md não encontrado ou vazio.", "warning")
            return True

        except requests.RequestException as e:
            status_code = e.response.status_code if hasattr(e, 'response') and e.response is not None else 'N/A'
            self._log_message(f"Erro ao buscar estrutura do repositório (Status: {status_code}): {e}", "error")
            if status_code == 404:
                 self._log_message(f"Verifique se o repo '{self.repo_owner}/{self.repo_name}' e a branch '{self.branch}' existem e são públicos (ou se o token é válido para repos privados).", "error")
            elif status_code == 403:
                 self._log_message("Erro 403: Limite de taxa da API GitHub ou token inválido/ausente. Tente mais tarde ou adicione/verifique seu GITHUB_TOKEN.", "error")
            return False
        except Exception as e:
             self._log_message(f"Erro inesperado ao processar estrutura: {e}", "error")
             return False

    def _fetch_file_content_by_sha(self, sha, file_path):
        """Busca e decodifica o conteúdo do arquivo usando o SHA, com limite aumentado."""
        api_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/git/blobs/{sha}"
        try:
            response = requests.get(api_url, headers=self.headers)
            response.raise_for_status()
            content_data = response.json()

            if content_data.get("encoding") == "base64" and content_data.get("content"):
                # Tratamento de erro na decodificação
                try:
                    decoded_content = base64.b64decode(content_data["content"]).decode('utf-8', errors='replace')
                except Exception as decode_err:
                    self._log_message(f"Erro ao decodificar {file_path}: {decode_err}. Pulando arquivo.", "warning")
                    return None # Falha na decodificação

                # --- Limite de caracteres por arquivo aumentado ---
                limited_content = decoded_content[:self.max_chars_per_file_read]
                if len(decoded_content) > self.max_chars_per_file_read:
                     # Log menos verboso para truncamento, só se for muito grande
                     if len(decoded_content) > self.max_chars_per_file_read * 1.5:
                         self._log_message(f"Conteúdo do arquivo '{file_path}' truncado em {self.max_chars_per_file_read / 1000:.0f}k caracteres.", "warning")
                return limited_content
                # --------------------------------------------------------
            elif content_data.get("encoding") != "base64":
                 # Arquivos não-base64 (ex: binários grandes detectados como texto) podem não ter 'content' ou ter encoding diferente
                 self._log_message(f"Arquivo '{file_path}' não está em base64 (encoding: {content_data.get('encoding', 'N/A')}). Pulando conteúdo.", "warning")
                 return None
            else:
                 # Pode ser um arquivo vazio
                 # self._log_message(f"Arquivo '{file_path}' tem resposta base64 mas sem conteúdo.", "info")
                 return "" # Retorna vazio se não tiver conteúdo

        except requests.RequestException as e:
            status_code = e.response.status_code if hasattr(e, 'response') and e.response is not None else 'N/A'
            # Erro comum: blob muito grande para a API retornar
            if status_code == 403 and "too large" in e.response.text.lower():
                 self._log_message(f"Arquivo '{file_path}' é muito grande para a API do GitHub retornar o conteúdo via blob SHA. Pulando.", "warning")
            else:
                 self._log_message(f"Erro {status_code} ao buscar conteúdo do arquivo '{file_path}': {e}", "warning")
            return None
        except json.JSONDecodeError as e:
            self._log_message(f"Erro ao decodificar JSON da resposta para '{file_path}': {e}", "warning")
            return None
        except Exception as e:
            self._log_message(f"Erro inesperado ao buscar conteúdo do arquivo '{file_path}': {e}", "warning")
            return None

    def analyze_repository(self, progress_callback=None):
        """Analisa o repositório, prepara dados e **avalia a complexidade**."""
        if not self.code_files and not self.readme_content:
            self._log_message("Nenhum arquivo de código ou README para analisar.", "warning")
            return False
        if progress_callback: progress_callback(0.55, "Analisando estrutura e conteúdo...") # Progresso após leitura
        self._log_message("Analisando conteúdo do repositório...", "info")
        self._extract_repo_summary()
        self._analyze_code_structure()
        self._identify_key_components()
        # --- NOVA ETAPA: Avaliar complexidade ---
        self._assess_repository_complexity()
        # -----------------------------------------
        self._log_message("Análise inicial concluída.", "success")
        return True

    def _extract_repo_summary(self):
        """Extrai informações sumárias do README."""
        self.repo_summary["title"] = f"{self.repo_name}"
        self.repo_summary["owner"] = self.repo_owner
        self.repo_summary["url"] = self.repo_url
        self.repo_summary["branch"] = self.branch

        if self.readme_content:
            # Tenta extrair título do H1
            match_title = re.search(r'^#\s+(.*?)\n', self.readme_content, re.MULTILINE)
            if match_title:
                self.repo_summary["title"] = match_title.group(1).strip()

            # Limpa badges comuns no início
            content_after_title = re.sub(r'^#\s+.*?\n', '', self.readme_content, count=1)
            content_cleaned = re.sub(r'^\[!\[.*?\]\(.*?\)\]\(.*?\)\s*?\n?', '', content_after_title, flags=re.MULTILINE)
            content_cleaned = re.sub(r'^\<p align="center"\>.*?\</p\>\s*', '', content_cleaned, flags=re.IGNORECASE | re.DOTALL) # Limpa blocos <p align="center">

            # Tenta pegar a primeira descrição significativa
            match_desc = re.search(r'^\s*(.+?)\n(\n|\s*#|$)', content_cleaned, re.DOTALL | re.MULTILINE)

            if match_desc:
                desc = match_desc.group(1).strip()
                # Limpa múltiplos espaços/newlines dentro da descrição
                desc_cleaned = ' '.join(desc.split())
                self.repo_summary["description"] = textwrap.shorten(desc_cleaned, width=500, placeholder="...")
            else:
                 # Fallback: pega as primeiras linhas não vazias
                 lines = [line for line in content_cleaned.strip().splitlines() if line.strip()][:5]
                 fallback_desc = ' '.join(lines)
                 self.repo_summary["description"] = textwrap.shorten(fallback_desc, width=500, placeholder="...")
        else:
            self.repo_summary["description"] = f"Um repositório GitHub de {self.repo_owner} sem README.md detalhado."

        self._log_message(f"Título: {self.repo_summary['title']}", "info")
        self._log_message(f"Descrição: {self.repo_summary['description']}", "info")

    def _analyze_code_structure(self):
        """Analisa os arquivos de código e sua estrutura."""
        self.repo_summary["languages"] = {}
        self.repo_summary["file_count"] = len(self.code_files)
        total_code_files_with_content = sum(1 for f in self.code_files if f.get("content") is not None) # Checa se content existe (mesmo que vazio)

        for file in self.code_files:
            if file["extension"]:
                # Trata extensões como '.dockerfile' ou '.yml'
                ext = file["extension"][1:] if file["extension"] != '.dockerfile' else 'dockerfile'
                ext = 'yaml' if ext == 'yml' else ext
                self.repo_summary["languages"][ext] = self.repo_summary["languages"].get(ext, 0) + 1

        # Ordena por contagem descendente
        self.repo_summary["languages"] = dict(sorted(self.repo_summary["languages"].items(), key=lambda item: item[1], reverse=True))
        self._log_message(f"Linguagens detectadas: {self.repo_summary['languages']}", "info")
        self._log_message(f"{total_code_files_with_content} arquivos tiveram conteúdo carregado para análise (total de {self.total_code_chars_collected / 1000:.1f}k caracteres).", "info")

    def _identify_key_components(self):
        """Identifica componentes chave e arquivos importantes (heurística simples)."""
        key_files = []
        # Patterns melhorados e expandidos
        patterns = {
            'config': ['config', 'settings', 'conf', 'manifest', 'values.yaml', 'chart.yaml', '.env', 'env.', 'properties', 'application.yml', 'bootstrap.yml', 'appsettings.json'],
            'entry': ['main', 'app', 'index', 'server', 'run', '__main__', 'wsgi', 'asgi', 'program.cs', 'startup.cs'],
            'build': ['package.json', 'requirements.txt', 'pom.xml', 'build.gradle', 'dockerfile', 'makefile', 'setup.py', 'go.mod', 'cargo.toml', 'composer.json', '.csproj', '.sln', 'rakefile'],
            'docs': ['contributing.md', 'license', 'code_of_conduct.md', 'changelog.md', 'readme.md'], # Adicionado readme aqui também
            'test': ['test_', '_test', '.spec.', '.test.'], # Mantido simples
            'workflow': ['.github/workflows', '.gitlab-ci.yml', 'azure-pipelines.yml', 'jenkinsfile'],
            'ui_framework_config': ['vite.config.', 'webpack.config.', 'next.config.', 'nuxt.config.', 'tailwind.config.', 'postcss.config.'],
            'orm_migration': ['migration', 'alembic', 'prisma/schema.prisma'],
            'routing': ['routes.', 'urls.py', 'controller', 'handler'],
        }
        # Arquivos/pastas comuns a serem verificados por existência
        common_checks = {
            'entry/framework': ['manage.py', 'artisan'],
            'docs': ['docs/', 'doc/'],
            'examples': ['examples/', 'example/'],
            'scripts': ['scripts/'],
            'src_dir': ['src/'],
            'lib_dir': ['lib/'],
            'app_dir': ['app/'],
            'components_dir': ['components/', 'ui/'],
            'api_dir': ['api/', 'server/'],
        }

        file_paths_lower = {f["path"].lower(): f["path"] for f in self.code_files}

        # 1. Checar por padrões de nome/extensão
        for file in self.code_files:
            fn_lower = file["name"].lower()
            path_lower = file["path"].lower()
            matched_type = None

            for type, terms in patterns.items():
                for term in terms:
                    # Verifica se o nome do arquivo começa com o termo, é igual ao termo, ou se a extensão casa (ex: config.js)
                    if fn_lower.startswith(term) or fn_lower == term or os.path.splitext(fn_lower)[0] == term:
                        # Exceção: evitar 'test' em nomes como 'latest.txt'
                        if type == 'test' and not ('test' in path_lower.split('/') or 'spec' in path_lower.split('/')):
                             continue
                        matched_type = type
                        break
                    # Verifica se o termo está contido no path (útil para workflows, tests, routing)
                    if term in path_lower and type in ['workflow', 'test', 'routing', 'orm_migration']:
                         # Refinamento para testes (deve estar numa pasta chamada test/tests/spec)
                         if type == 'test' and not any(p in path_lower.split('/') for p in ['test', 'tests', 'spec']):
                             continue
                         matched_type = type
                         break
                if matched_type: break

            if matched_type:
                 key_files.append({"path": file["path"], "type": matched_type})

        # 2. Checar por existência de arquivos/pastas comuns
        checked_common_paths = set()
        for type, paths in common_checks.items():
            for path_pattern in paths:
                 # Verifica se começa com (para pastas) ou é igual (para arquivos)
                 found_path = None
                 if path_pattern.endswith('/'): # É uma pasta
                     prefix = path_pattern
                     for fp_lower, fp_orig in file_paths_lower.items():
                         if fp_lower.startswith(prefix) and prefix not in checked_common_paths:
                            found_path = prefix # Marca a pasta como encontrada
                            checked_common_paths.add(prefix)
                            break
                 else: # É um arquivo
                     if path_pattern in file_paths_lower and path_pattern not in checked_common_paths:
                         found_path = file_paths_lower[path_pattern] # Pega o path original
                         checked_common_paths.add(path_pattern)


                 if found_path and not any(kf['path'] == found_path for kf in key_files):
                     # Adiciona se encontrou e ainda não está na lista
                     key_files.append({"path": found_path, "type": type})


        # Limita a lista enviada no sumário (mas a IA terá acesso a mais arquivos no contexto)
        # Prioriza tipos mais importantes se a lista for muito grande
        key_files.sort(key=lambda x: (
            0 if x['type'] in ['entry', 'build', 'config', 'ui_framework_config'] else
            1 if x['type'] in ['routing', 'orm_migration', 'workflow'] else
            2 if x['type'] == 'readme.md' else
            3 # Outros
        ))
        self.repo_summary["key_files_guess"] = key_files[:30] # Aumenta um pouco o limite do sumário
        self._log_message(f"Possíveis arquivos/pastas chave identificados ({len(key_files)} total): {[f['path'] for f in self.repo_summary['key_files_guess']]}", "info")

    # --- NOVO MÉTODO ---
    def _assess_repository_complexity(self):
        """Avalia a complexidade do repositório com base em métricas coletadas."""
        num_files = len(self.code_files)
        num_langs = len(self.repo_summary.get("languages", {}))
        readme_len = len(self.readme_content)
        code_chars = self.total_code_chars_collected

        score = 0

        # Pontuação baseada em caracteres de código (com logaritmo para não explodir)
        if code_chars > 1000: # Evita log(0) ou valores muito pequenos
             # Normaliza um pouco usando log, pondera mais
             score += math.log10(code_chars / 1000) * 2.5

        # Pontuação baseada no número de arquivos (com logaritmo)
        if num_files > 10:
            score += math.log10(num_files / 10) * 1.5

        # Pontuação por número de linguagens
        score += max(0, num_langs - 1) * 0.5 # Bônus por cada linguagem além da primeira

        # Bônus/Penalidade por README
        if readme_len < 200 and code_chars > 10000: # Penalidade leve se não tem README mas tem código
            score -= 0.5
        elif readme_len > 5000: # Bônus leve por README detalhado
            score += 0.5

        # Bônus se a API truncou a lista de arquivos (indica repo muito grande)
        # (Precisaria checar o 'truncated' flag da resposta da API, mas não armazenamos ele globalmente)
        # if self.is_truncated: score += 2

        # Classificação final baseada no score
        # Ajuste estes limiares de score conforme observação
        if score < 4.0:
             self.complexity_level = 'simple'
        elif score > 9.0:
             self.complexity_level = 'complex'
        else:
             self.complexity_level = 'medium'

        self._log_message(f"Complexidade do repositório avaliada como: {self.complexity_level.upper()} (Score: {score:.2f})", "info")
        self._log_message(f"Critérios: Arquivos={num_files}, Chars Código={code_chars/1000:.1f}k, Linguagens={num_langs}, README={readme_len} chars", "info")

    def generate_podcast_script(self, progress_callback=None):
        """
        Gera o script do podcast usando o modelo Gemini.
        **Adapta o prompt para pedir um script conciso ou detalhado com base na complexidade.**
        """
        # Assegura que a análise (e avaliação de complexidade) foi feita
        if not self.repo_summary:
             if not self.analyze_repository(progress_callback):
                 self._log_message("Falha na análise inicial, não é possível gerar script.", "error")
                 return self._generate_stub_podcast_script("Falha ao analisar o repositório.")

        # Determina qual prompt usar com base na complexidade
        if self.complexity_level == 'complex':
            prompt_text = self._get_detailed_prompt()
            log_prefix = "DETALHADO"
            progress_stage = 0.7 # Inicia um pouco mais tarde se for detalhado
            progress_msg = f"Gerando roteiro DETALHADO com {self.gemini_model_name}..."
        else: # Simple ou Medium usarão o conciso
            prompt_text = self._get_concise_prompt()
            log_prefix = "CONCISO"
            progress_stage = 0.75
            progress_msg = f"Gerando roteiro CONCISO com {self.gemini_model_name}..."

        self._log_message(f"Gerando script {log_prefix} do podcast com {self.gemini_model_name}...", "info")

        if not self.gemini_api_key:
            self._log_message("API Key da Gemini não configurada. Gerando script de exemplo.", "error")
            return self._generate_stub_podcast_script()

        # --- Preparação de dados para IA (mesma lógica de antes, mas com limites maiores) ---
        if progress_callback: progress_callback(progress_stage * 0.8, "Preparando dados para IA...") # Ajuste no progresso

        key_file_paths = [kf['path'] for kf in self.repo_summary.get('key_files_guess', [])]
        # Prioriza arquivos chave E arquivos com mais conteúdo (indicativo de importância)
        files_with_content = sorted(
            [f for f in self.code_files if f.get('content') is not None],
             key=lambda f: (0 if f['path'] in key_file_paths else 1, -len(f.get('content', '')), f['path']) # Negativo do len para ordenar por tamanho desc
        )

        code_snippets_for_ai = []
        total_code_chars_to_send = 0

        self._log_message(f"Tentando incluir até ~{self.max_total_code_chars_for_ai / 1000:.0f}k caracteres de código no prompt para a IA.", "info")

        for file in files_with_content:
            content_to_add = file.get('content', '') # Pega conteúdo (pode ser vazio)
            # Verifica limite total E limite por arquivo (redundante, mas seguro)
            if total_code_chars_to_send + len(content_to_add) <= self.max_total_code_chars_for_ai and len(content_to_add) <= self.max_chars_per_file_read:
                code_snippets_for_ai.append({
                    "path": file["path"],
                    "content": content_to_add
                })
                total_code_chars_to_send += len(content_to_add)
            else:
                # Para o loop se atingir o limite total
                self._log_message(f"Limite total de {self.max_total_code_chars_for_ai / 1000:.0f}k caracteres de código para IA atingido ou arquivo individual muito grande. {len(code_snippets_for_ai)}/{len(files_with_content)} arquivos com conteúdo incluídos no prompt.", "warning")
                break

        self._log_message(f"Total de caracteres de código efetivamente incluídos para a IA: {total_code_chars_to_send / 1000:.1f}k", "info")

        readme_for_ai = self.readme_content[:self.max_readme_chars_for_ai]
        if len(self.readme_content) > self.max_readme_chars_for_ai:
             self._log_message(f"Conteúdo do README truncado em {self.max_readme_chars_for_ai / 1000:.0f}k caracteres para a IA.", "warning")

        all_files_list_for_ai = [f["path"] for f in self.code_files][:self.max_file_list_for_ai]
        if len(self.code_files) > self.max_file_list_for_ai:
             self._log_message(f"Lista de arquivos truncada em {self.max_file_list_for_ai} nomes para a IA.", "warning")

        # O dicionário enviado à IA NÃO contém a análise de complexidade.
        # A complexidade só influencia QUAL prompt é enviado.
        repo_data_for_ai = {
            "summary": self.repo_summary,
            "readme_content": readme_for_ai,
            "code_files_analyzed_count": len(code_snippets_for_ai),
            "total_code_chars_analyzed": total_code_chars_to_send,
            "code_files_analyzed": code_snippets_for_ai,
            "all_files_list": all_files_list_for_ai
        }
        # -----------------------------------------------------------------

        try:
            if progress_callback: progress_callback(progress_stage, progress_msg)

            # --- Mesma lógica de chamada da API ---
            # ATENÇÃO: Modelos como 1.5 Pro podem lidar com contextos MUITO maiores.
            # Certifique-se de que sua cota e limites da API comportam o volume de dados.
            model = genai.GenerativeModel(self.gemini_model_name)
            prompt_chars = len(prompt_text)
            # Estima o tamanho dos dados em JSON. Pode ser grande.
            try:
                 data_str = json.dumps(repo_data_for_ai)
                 data_chars = len(data_str)
            except Exception as json_e:
                 self._log_message(f"Erro ao serializar dados para IA: {json_e}. Usando estimativa.", "warning")
                 data_chars = total_code_chars_to_send + len(readme_for_ai) + len(str(self.repo_summary)) + 10000 # Estimativa grosseira

            total_input_chars_est = prompt_chars + data_chars
            self._log_message(f"Enviando prompt ({log_prefix}) + dados (~{total_input_chars_est / 1000:.0f}k caracteres estimados) para {self.gemini_model_name}...", "info")

            # Envia o prompt escolhido e os dados
            response = model.generate_content(
                 [prompt_text, json.dumps(repo_data_for_ai)], # Envia como partes separadas
                 generation_config=self.generation_config,
                 safety_settings=self.safety_settings
            )
            # --------------------------------------

            # --- Mesma lógica de tratamento de resposta e erro ---
            if not response.candidates:
                 # Tenta obter mais detalhes sobre o bloqueio
                 block_reason = "Desconhecido"
                 safety_ratings_str = "N/A"
                 try:
                     if response.prompt_feedback:
                        block_reason = response.prompt_feedback.block_reason
                        if response.prompt_feedback.safety_ratings:
                             safety_ratings_str = str(response.prompt_feedback.safety_ratings)
                 except Exception as e:
                     self._log_message(f"Erro ao acessar detalhes do feedback: {e}", "warning")

                 self._log_message(f"Geração bloqueada pela API Gemini ({self.gemini_model_name}). Razão: {block_reason}", "error")
                 self._log_message(f"Classificações de segurança do prompt: {safety_ratings_str}", "warning")
                 return self._generate_stub_podcast_script(f"Erro: Conteúdo bloqueado pela política de segurança da IA ({self.gemini_model_name}). Razão: {block_reason}")

            # Extrai o texto da primeira (e única) candidata
            script = response.text

            # Adiciona rodapé padrão
            script += f"""
            \n\n---\n
            [VINHETA DE ENCERRAMENTO]\n
            Roteiro {log_prefix} gerado por Explica Código (usando {self.gemini_model_name}).
            Repositório analisado: {self.repo_url}
            Complexidade estimada: {self.complexity_level.upper()}
            """

            self._log_message(f"Script {log_prefix} do podcast gerado com sucesso!", "success")
            if progress_callback: progress_callback(0.9, f"Roteiro {log_prefix.lower()} gerado!")
            return script.strip()

        # --- Tratamento de Erros da API Gemini ---
        except genai.types.BlockedPromptException as e:
             self._log_message(f"Geração bloqueada (BlockedPromptException) pela API Gemini ({self.gemini_model_name}).", "error")
             # Tentar extrair detalhes, embora possa não haver na exceção diretamente
             return self._generate_stub_podcast_script(f"Erro: Conteúdo bloqueado pela política de segurança da IA ({self.gemini_model_name}).")
        except genai.types.StopCandidateException as e:
             self._log_message(f"Geração interrompida (StopCandidateException) pela API Gemini ({self.gemini_model_name}). Pode ser devido a políticas de segurança na *resposta*.", "error")
             # A resposta parcial pode estar em e.candidate, mas é arriscado usar
             return self._generate_stub_podcast_script(f"Erro: Geração interrompida pela IA ({self.gemini_model_name}), possivelmente por segurança no conteúdo gerado.")
        except requests.exceptions.RequestException as e: # Erros de rede/HTTP
             self._log_message(f"Erro de rede/HTTP ao comunicar com a API Gemini: {e}", "error")
             return self._generate_stub_podcast_script(f"Erro de comunicação com a API: {e}")
        except Exception as e: # Outros erros genéricos
             self._log_message(f"Erro inesperado ao gerar script com {self.gemini_model_name}: {e}", "error")
             # Tratamento específico para erros comuns
             if "429" in str(e) or "ResourceExhaustedError" in str(e) or "rate limit" in str(e).lower():
                 self._log_message("Erro 429: Limite de taxa da API Gemini atingido (RPM/TPM). Tente novamente mais tarde ou verifique sua cota.", "error")
                 error_detail = "Limite de taxa da API Gemini atingido."
             elif "API key not valid" in str(e) or "permission denied" in str(e).lower():
                  self._log_message("Erro: API Key da Gemini inválida ou sem permissão.", "error")
                  error_detail = "API Key inválida ou sem permissão."
             elif "invalid json" in str(e).lower():
                  self._log_message("Erro: Problema ao formatar os dados enviados para a IA (JSON inválido). Verifique a estrutura dos dados.", "error")
                  error_detail = "Erro interno na formatação dos dados."
             else:
                  error_detail = f"Erro inesperado na IA: {e}"

             import traceback
             self._log_message(traceback.format_exc(), "error") # Log completo do traceback para debug
             return self._generate_stub_podcast_script(f"Erro na comunicação com o modelo {self.gemini_model_name}: {error_detail}")

    # --- NOVO MÉTODO PARA PROMPT CONCISO ---
    def _get_concise_prompt(self):
        """Retorna o texto do prompt para gerar um roteiro CONCISO."""
        return f"""
        Você é um Gerador de Roteiros de Podcast chamado 'Explica Código'. Sua missão é criar um roteiro de podcast **conciso, objetivo e informativo** em português do Brasil (pt-BR) explicando os **pontos ESSENCIAIS** de um repositório do GitHub para desenvolvedores juniores a plenos que buscam uma **visão geral rápida e de qualidade**.

        **Repositório Alvo:** (Informações básicas)
        - URL: {self.repo_url}
        - Proprietário: {self.repo_owner}
        - Nome: {self.repo_name}
        - Branch: {self.branch}

        **Dados Coletados do Repositório (Contexto Fornecido):**
        Você recebeu um volume considerável de informações (README, lista completa de arquivos, conteúdo detalhado de muitos arquivos). **Use este contexto para SINTETIZAR e extrair os aspectos MAIS IMPORTANTES.** Não detalhe tudo, foque no que é crucial para entender o projeto rapidamente.

        **Seu Roteiro de Podcast (Aproximadamente 5-8 minutos de fala - CONCISO E DIRETO)**

        **Instruções IMPORTANTES:**
        *   **NÃO inclua no roteiro final nenhuma instrução sobre como você criou o roteiro, nem menções sobre a complexidade do repositório ou o tempo estimado.** Apenas o roteiro em si.
        *   O público alvo são desenvolvedores Jr/Pleno, então use termos técnicos, mas explique conceitos complexos brevemente se necessário.
        *   O objetivo é dar uma visão geral rápida e útil.

        **Estrutura Sugerida para o Roteiro:**
        1.  **[VINHETA DE ABERTURA]**
        2.  **Introdução:** Apresente o podcast e o repositório de hoje (nome, dono).
        3.  **Propósito Principal:** Qual problema o projeto resolve? Qual seu objetivo central? (Seja direto, baseado no README e análise geral).
        4.  **Visão Geral da Arquitetura:** Como o projeto está estruturado em *alto nível*? Mencione o principal padrão (se evidente) ou os diretórios/módulos mais importantes e como se conectam *brevemente*. **Não liste todos os arquivos.**
        5.  **Tecnologias Chave:** Liste as 2-4 tecnologias *mais importantes* (linguagens, frameworks, libs significativas) e explique *sucintamente* seu papel essencial no projeto.
        6.  **Ponto(s) de Destaque:** Escolha **UM ou DOIS** aspectos interessantes ou cruciais do código/implementação (pode ser um algoritmo chave simplificado, uma boa prática notável, uma configuração importante ou um desafio técnico principal abordado) e explique-o de forma acessível. **Use o código fornecido como base, mas resuma a ideia.**
        7.  **[VINHETA DE TRANSIÇÃO]** (Opcional, se mudar muito de assunto)
        8.  **Para o Aprendiz (Dicas Rápidas):**
            *   Qual o melhor **ponto de partida** para explorar o código? (Ex: um arquivo principal, a pasta de componentes, a documentação).
            *   Que **conceito chave** ou **boa prática** o dev pode aprender observando este projeto?
        9.  **Conclusão:** Recapitule brevemente e incentive a exploração do repo.
        10. **[VINHETA DE ENCERRAMENTO]** (Será adicionada automaticamente no final, mas você pode indicar onde terminaria a fala).

        **Formato do Roteiro:**
        *   Use marcadores como [VINHETA DE ABERTURA], [VINHETA DE TRANSIÇÃO], etc. nos locais apropriados.
        *   Indique o locutor como "LOCUTOR:".
        *   Linguagem clara e objetiva. Evite jargões excessivos sem explicação rápida.
        *   **Seja seletivo.** Foque na qualidade da informação essencial, não na quantidade. A concisão é chave.
        *   **Priorize a precisão técnica** ao resumir, baseada no código e dados fornecidos.

        **Comece o roteiro imediatamente após esta linha, com a vinheta:**
        [VINHETA DE ABERTURA]

        LOCUTOR: Olá, coders! Bem-vindos ao Explica Código, o podcast que descomplica repositórios do GitHub para você...
        """

    # --- NOVO MÉTODO PARA PROMPT DETALHADO ---
    def _get_detailed_prompt(self):
        """Retorna o texto do prompt para gerar um roteiro DETALHADO."""
        return f"""
        Você é um Gerador de Roteiros de Podcast chamado 'Explica Código'. Sua missão é criar um roteiro de podcast **detalhado, aprofundado e didático** em português do Brasil (pt-BR) explicando um repositório **complexo** do GitHub para desenvolvedores juniores a plenos que desejam **entender a fundo** sua estrutura, funcionamento e nuances.

        **Repositório Alvo:** (Informações básicas)
        - URL: {self.repo_url}
        - Proprietário: {self.repo_owner}
        - Nome: {self.repo_name}
        - Branch: {self.branch}

        **Dados Coletados do Repositório (Contexto Extenso Fornecido):**
        Você recebeu um GRANDE volume de informações (README completo, lista extensiva de arquivos, e o conteúdo detalhado de MUITOS arquivos de código e configuração). **Sua tarefa é USAR essa riqueza de detalhes para criar um roteiro que explore a complexidade do projeto.** Vá além da superfície.

        **Seu Roteiro de Podcast (Aproximadamente 10-15 minutos de fala - DETALHADO E EXPLORATÓRIO)**

        **Instruções IMPORTANTES:**
        *   **NÃO inclua no roteiro final nenhuma instrução sobre como você criou o roteiro, nem menções sobre a complexidade do repositório ou o tempo estimado.** Apenas o roteiro em si.
        *   O público alvo são desenvolvedores Jr/Pleno. Use termos técnicos apropriados, mas **explique conceitos mais avançados ou específicos do domínio** do projeto.
        *   O objetivo é fornecer um entendimento mais profundo. **Não tenha medo de entrar em detalhes técnicos relevantes.**

        **Estrutura Sugerida para o Roteiro Detalhado:**
        1.  **[VINHETA DE ABERTURA]**
        2.  **Introdução:** Apresente o podcast e o repositório (nome, dono), mencionando que é um projeto mais robusto/complexo.
        3.  **Propósito e Contexto:** Qual problema o projeto resolve? Qual seu objetivo principal? Existe algum contexto específico (ex: parte de um ecossistema maior, pesquisa acadêmica)?
        4.  **Arquitetura e Organização do Código:**
            *   Descreva a estrutura de pastas principal de forma mais detalhada. Quais são os módulos/componentes chave e suas responsabilidades?
            *   Qual o padrão arquitetural predominante (MVC, Microservices, Hexagonal, etc.)? Como ele se manifesta no código?
            *   Como os dados fluem pelo sistema (se aplicável)? Existe um fluxo principal a ser destacado?
        5.  **Tecnologias e Dependências:**
            *   Liste as tecnologias *principais* (linguagens, frameworks, bancos de dados, bibliotecas críticas).
            *   Explique *por que* algumas dessas tecnologias foram provavelmente escolhidas (suas vantagens no contexto do projeto).
            *   Mencione quaisquer dependências *incomuns ou particularmente importantes*.
        6.  **[VINHETA DE TRANSIÇÃO]**
        7.  **Exploração Técnica Aprofundada (Escolha 2-3 pontos):**
            *   Analise um **algoritmo ou lógica de negócio central**. Explique como funciona, baseado no código fornecido.
            *   Discuta uma **configuração complexa** (ex: build, deploy, infraestrutura como código) e seu propósito.
            *   Destaque um **padrão de design** bem aplicado ou uma solução inteligente para um desafio técnico.
            *   Analise como os **testes** estão estruturados ou uma estratégia de teste interessante (se houver dados).
            *   Comente sobre aspectos de **performance, segurança ou escalabilidade** que parecem ter sido considerados no código.
        8.  **[VINHETA DE TRANSIÇÃO]** (Opcional)
        9.  **Para o Contribuidor/Explorador:**
            *   Quais são os **melhores pontos de partida** para entender diferentes partes do sistema? (Ex: "Para entender a API, comece por `api/routes.py`"; "Para a lógica de UI, veja `src/components/`").
            *   Existem **guias de contribuição ou documentação** importantes a serem lidos (mencione o `CONTRIBUTING.md` se relevante)?
            *   Quais **conceitos avançados** ou **ferramentas específicas** um desenvolvedor pode aprender ao estudar este projeto?
        10. **Conclusão:** Recapitule os pontos chave da análise profunda e reforce o valor de explorar repositórios complexos.
        11. **[VINHETA DE ENCERRAMENTO]** (Será adicionada automaticamente no final).

        **Formato do Roteiro:**
        *   Use marcadores como [VINHETA DE ABERTURA], [VINHETA DE TRANSIÇÃO], etc.
        *   Indique o locutor como "LOCUTOR:".
        *   Seja claro, mas permita-se ser mais técnico e detalhado do que no roteiro conciso.
        *   **Use o código e os dados fornecidos extensivamente** para basear suas explicações técnicas. Cite nomes de arquivos ou funções chave quando relevante para ilustrar um ponto.
        *   Mantenha um fluxo lógico e didático.

        **Comece o roteiro imediatamente após esta linha, com a vinheta:**
        [VINHETA DE ABERTURA]

        LOCUTOR: Olá, coders! Bem-vindos ao Explica Código. Hoje, vamos mergulhar fundo em um repositório mais robusto...
        """

    # --- MÉTODOS DE EXEMPLO E GERAÇÃO DE ÁUDIO (sem alterações significativas) ---

    def _generate_stub_podcast_script(self, error_message="API indisponível ou erro na geração"):
        """Gera um script de exemplo quando a API falha."""
        self._log_message(f"Gerando script de exemplo devido a erro: {error_message}", "warning")
        repo_title = self.repo_summary.get("title", self.repo_name if self.repo_name else "N/A")
        repo_owner_name = self.repo_owner if self.repo_owner else "N/A"
        main_language = next(iter(self.repo_summary.get("languages", {"código"}).keys()), "código")
        repo_desc = self.repo_summary.get('description', 'Não foi possível carregar a descrição.')
        repo_link = self.repo_url if self.repo_url else "URL não disponível"

        # Tenta incluir o nome do modelo que falhou, se disponível
        model_name_mention = f" (tentativa com {self.gemini_model_name})" if hasattr(self, 'gemini_model_name') and self.gemini_model_name else ""

        return f"""
        [VINHETA DE ABERTURA]

        LOCUTOR: Olá! Bem-vindos ao Explica Código! Infelizmente, hoje encontramos um problema técnico ao tentar gerar a análise detalhada do repositório{model_name_mention}. A mensagem de erro foi: "{error_message}".

        LOCUTOR: Mesmo sem a análise da IA, vamos dar uma olhada nas informações básicas que conseguimos coletar.

        [VINHETA DE TRANSIÇÃO]

        LOCUTOR: O repositório que íamos explorar é o '{repo_title}', mantido por '{repo_owner_name}'. Pelo que vimos, ele parece ser escrito principalmente em '{main_language}'.

        LOCUTOR: A descrição que encontramos sugere que o objetivo do projeto é: {repo_desc}

        LOCUTOR: Como não conseguimos gerar o roteiro completo, a dica de hoje é mais genérica: ao explorar um novo repositório, sempre comece pelo README.md! Ele geralmente contém a visão geral, instruções de instalação e uso.

        LOCUTOR: Depois do README, procure por arquivos de configuração (como 'package.json', 'requirements.txt', 'pom.xml'), arquivos de ponto de entrada (como 'main.py', 'index.js', 'Program.cs') e a estrutura geral de pastas (como 'src', 'app', 'lib', 'tests', 'docs').

        LOCUTOR: Pedimos desculpas por não termos o episódio completo hoje. Recomendamos que você mesmo navegue pelo repositório no link que deixaremos na descrição, caso ele esteja disponível: {repo_link}

        [VINHETA DE ENCERRAMENTO]
        ---
        Roteiro de exemplo gerado devido a erro.
        Repositório: {repo_link}
        Erro: {error_message}
        """

    def generate_podcast_audio(self, script_text, lang='pt-br', vignette_path="background_music.mp3", vignette_duration_ms=5000, progress_callback=None):
        """
        Gera o áudio do podcast, inserindo uma vinheta musical curta
        nos locais indicados por marcadores no script.
        **Modificado para limpar melhor o texto antes de enviar ao TTS.**
        """
        self._log_message("Preparando áudio do podcast com vinhetas...", "info")
        if progress_callback:
            progress_callback(0.95, "Carregando vinheta e processando roteiro...")

        # --- Carregar e preparar a vinheta ---
        vignette = None # Inicializa como None
        if vignette_path and Path(vignette_path).exists():
            try:
                # Aumenta um pouco o fade out para suavizar
                vignette_full = AudioSegment.from_mp3(vignette_path)
                vignette = vignette_full[:vignette_duration_ms]
                fade_duration = min(1000, vignette_duration_ms // 4) # Fade out de até 1s
                vignette = vignette.fade_out(duration=fade_duration)
                self._log_message(f"Vinheta '{vignette_path}' carregada e preparada ({len(vignette)/1000:.1f}s).", "info")
            except FileNotFoundError:
                self._log_message(f"Arquivo de vinheta '{vignette_path}' não encontrado. Continuando sem vinhetas.", "warning")
                vignette_path = None # Garante que não tentará usar depois
            except Exception as e:
                self._log_message(f"Erro ao carregar ou processar vinheta '{vignette_path}': {e}. Verifique o arquivo e a instalação do ffmpeg/librosa. Continuando sem vinhetas.", "warning")
                vignette_path = None
        else:
             self._log_message(f"Caminho da vinheta não fornecido ou inválido ('{vignette_path}'). Continuando sem vinhetas.", "warning")
             vignette_path = None

        # --- Processar o roteiro e gerar áudio ---
        final_segments = []
        # Marcadores que devem inserir a vinheta (se carregada)
        vignette_markers = [
            '[VINHETA DE ABERTURA]',
            '[VINHETA DE TRANSIÇÃO]',
            '[VINHETA DE ENCERRAMENTO]'
        ]
        # Marcadores que devem ser REMOVIDOS do texto falado
        markers_to_remove = vignette_markers + [
            '[MÚSICA SUAVE DE FUNDO]', # Exemplo de outros marcadores a remover
            '[SECTION BREAK]',
        ]

        # Padrão para encontrar QUALQUER marcador entre colchetes
        marker_pattern = r'(\[.*?\])'
        parts = [p.strip() for p in re.split(marker_pattern, script_text) if p and p.strip()]

        if not parts:
             self._log_message("Roteiro vazio ou sem partes reconhecíveis após divisão.", "error")
             return None

        total_parts = len(parts)
        processed_parts = 0

        try:
            for part in parts:
                processed_parts += 1
                # Calcula progresso da etapa de áudio (últimos 5%)
                current_progress = 0.95 + (0.05 * (processed_parts / total_parts))
                if progress_callback:
                     progress_callback(current_progress, f"Processando áudio parte {processed_parts}/{total_parts}...")

                is_vignette_marker = part in vignette_markers
                is_marker_to_remove = part in markers_to_remove

                if is_vignette_marker and vignette:
                    self._log_message(f"Adicionando vinheta para o marcador: {part}", "info")
                    final_segments.append(vignette)
                elif is_marker_to_remove:
                    # Apenas ignora o marcador se ele deve ser removido e não é de vinheta (ou a vinheta falhou)
                    self._log_message(f"Removendo marcador: {part}", "info")
                    continue
                elif part.startswith('[') and part.endswith(']'):
                     # É um marcador desconhecido, loga e ignora
                     self._log_message(f"Ignorando marcador desconhecido: {part}", "warning")
                     continue
                else: # É um bloco de texto para falar
                    # 1. Limpeza inicial (metadados no final, linhas vazias)
                    text_to_speak = re.sub(r'---\n.*', '', part, flags=re.DOTALL).strip()
                    text_to_speak = "\n".join(line for line in text_to_speak.splitlines() if line.strip())

                    if not text_to_speak:
                        continue # Pula se não sobrou texto útil

                    # 2. Limpeza específica para TTS (REMOVER MARCADORES INDESEJADOS)
                    # Remove **LOCUTOR:** ou LOCUTOR: (com ou sem asteriscos e espaços) no início das linhas
                    text_to_speak = re.sub(r'^\s*(\*\*)*(LOCUTOR:|HOST:)(\*\*)*\s*', '', text_to_speak, flags=re.IGNORECASE | re.MULTILINE)
                    # Remove asteriscos de negrito/itálico (**) ou (*)
                    text_to_speak = re.sub(r'\*(\*?)(.*?)\1\*', r'\2', text_to_speak) # Remove **texto** ou *texto* deixando só 'texto'
                    # Remove crases (backticks) usadas para código inline
                    text_to_speak = re.sub(r'`', '', text_to_speak)
                    # Remove cabeçalhos Markdown (#, ##, etc.) no início das linhas
                    text_to_speak = re.sub(r'^\s*#+\s+', '', text_to_speak, flags=re.MULTILINE)
                    # Remove possíveis marcadores de lista restantes (- , * , + ) no início de linha se não foram convertidos em frase
                    text_to_speak = re.sub(r'^\s*[-*+]\s+', '', text_to_speak, flags=re.MULTILINE)
                    # Remove links markdown [texto](url), mantendo só o texto
                    text_to_speak = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text_to_speak)

                    # Limpa espaços extras que podem ter sido deixados pelas substituições
                    text_to_speak = re.sub(r'\s+', ' ', text_to_speak).strip()
                    # Tenta corrigir pontuação comum antes de vírgulas/pontos
                    text_to_speak = re.sub(r'\s([?.!,:])', r'\1', text_to_speak)

                    if text_to_speak:
                        self._log_message(f"Gerando fala para: '{textwrap.shorten(text_to_speak, 80)}...'", "info")
                        try:
                            # --- Geração de Fala com gTTS ---
                            tts = gTTS(text=text_to_speak, lang=lang, slow=False)
                            speech_fp = io.BytesIO()
                            tts.write_to_fp(speech_fp)
                            speech_fp.seek(0)

                            # --- Carregamento com Pydub ---
                            try:
                                speech_segment = AudioSegment.from_mp3(speech_fp)
                                # Adiciona um pequeno silêncio após cada bloco de fala para melhor respiração
                                final_segments.append(speech_segment)
                                final_segments.append(AudioSegment.silent(duration=300)) # 300ms de silêncio
                                self._log_message(f"Segmento de fala gerado ({len(speech_segment)/1000:.1f}s).", "info")
                            except Exception as e_pydub:
                                 self._log_message(f"Erro ao carregar segmento de fala com pydub: {e_pydub}. Verifique ffmpeg. Pulando segmento.", "warning")
                                 # Tenta logar mais detalhes se for erro de decodificação
                                 if "decoder" in str(e_pydub).lower():
                                     self._log_message("Isso pode indicar um problema com a instalação do ffmpeg ou formato de áudio inesperado do gTTS.", "warning")
                                 continue # Pula este segmento

                        except Exception as e_tts:
                            self._log_message(f"Erro ao gerar fala para um segmento com gTTS: {e_tts}", "warning")
                            # Verifica erros comuns do gTTS
                            if "429 (Too Many Requests)" in str(e_tts):
                                self._log_message("gTTS retornou erro 429 - limite de requisições atingido. Tente novamente mais tarde.", "error")
                                # Poderia implementar um backoff aqui, mas por simplicidade, apenas avisamos.
                            continue # Pula este segmento

            # --- Combinar todos os segmentos ---
            if not final_segments:
                self._log_message("Nenhum segmento de áudio foi gerado.", "error")
                return None

            self._log_message("Combinando segmentos de fala e vinhetas...", "info")
            combined_audio = AudioSegment.empty()
            for segment in final_segments:
                 combined_audio += segment

            # Remove silêncio extra no final, se houver
            if len(combined_audio) > 0:
                combined_audio = combined_audio.strip_silence(silence_len=100, silence_thresh=combined_audio.dBFS - 16) # Ajuste threshold se necessário


            # --- Exportar o áudio final ---
            final_audio_fp = io.BytesIO()
            # Exporta com bitrate razoável para podcasts
            combined_audio.export(final_audio_fp, format="mp3", bitrate="128k")
            final_audio_fp.seek(0)

            self._log_message(f"Áudio final com vinhetas gerado com sucesso! Duração total: {len(combined_audio)/1000:.1f}s", "success")
            if progress_callback:
                 progress_callback(1.0, "Podcast pronto!")
            return final_audio_fp

        except Exception as e:
            self._log_message(f"Erro inesperado durante a geração/concatenação de áudio: {e}", "error")
            import traceback
            self._log_message(traceback.format_exc(), "error")
            return None


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

# Carrega variáveis de ambiente do .env (se existir)
load_dotenv()

class GitHubPodcastGenerator:
    """
    Ferramenta para gerar podcasts didáticos em áudio explicando repositórios do GitHub,
    focando em desenvolvedores juniores, usando a API Gemini e gTTS.
    Versão adaptada para um modelo hipotético de contexto longo (ex: 1M tokens).
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
        self.gemini_model_name = 'gemini-2.5-pro-exp-03-25' # ATENÇÃO: Modelo hipotético/experimental!
        self._log_message(f"Usando modelo Gemini: {self.gemini_model_name}", "warning")
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
        self.code_extensions = ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.h', '.go', '.rb', '.php', '.ts', '.jsx', '.tsx', '.ipynb', '.sh', '.md']

        # Armazena dados do repositório
        self.code_files = []
        self.readme_content = ""
        self.repo_summary = {}
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

        # --- NOVOS PARÂMETROS DE LIMITE (AUMENTADOS para modelo 1M tokens) ---
        # Limite alto, mas evita repositórios GIGANTES travarem tudo só na leitura
        self.max_files_to_read_content = 1000
        # Limite alto por arquivo (ex: ~100k caracteres)
        self.max_chars_per_file_read = 100000
        # Limite para o README enviado à IA (ex: ~200k caracteres)
        self.max_readme_chars_for_ai = 200000
        # Limite TOTAL de caracteres de CÓDIGO a serem enviados à IA (ex: 3 Milhões)
        # Deixa espaço para prompt, readme, lista de arquivos, resposta, etc., dentro dos ~4M totais de chars (1M tokens * ~4 chars/token).
        self.max_total_code_chars_for_ai = 3000000
        # Limite para a lista de nomes de arquivos enviada à IA
        self.max_file_list_for_ai = 2000
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

            if len(path_parts) > 3 and path_parts[2] == "tree":
                self.branch = path_parts[3]
            else:
                self.branch = self._get_default_branch()

            self._log_message(f"Repositório: {self.repo_owner}/{self.repo_name} (Branch: {self.branch})", "info")
            return True
        except ValueError as e:
            self._log_message(f"Erro ao analisar URL: {e}", "error")
            return False
        except Exception as e:
            self._log_message(f"Erro inesperado ao analisar URL: {e}", "error")
            return False

    def _get_default_branch(self):
        """Obtém a branch padrão do repositório via API do GitHub."""
        api_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}"
        try:
            response = requests.get(api_url, headers=self.headers)
            response.raise_for_status()
            repo_info = response.json()
            default_branch = repo_info.get("default_branch", "main")
            self._log_message(f"Branch padrão detectada: {default_branch}", "info")
            return default_branch
        except requests.RequestException as e:
            self._log_message(f"Não foi possível obter a branch padrão (usando 'main'): {e}", "warning")
            return "main"

    def fetch_repo_structure(self, progress_callback=None):
        """Busca a estrutura do repositório e arquivos importantes, com limites aumentados."""
        self._log_message("Buscando estrutura do repositório (modo contexto amplo)...", "info")
        self.code_files = []
        self.readme_content = ""

        api_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/git/trees/{self.branch}?recursive=1"

        try:
            response = requests.get(api_url, headers=self.headers)
            response.raise_for_status()
            tree_data = response.json()

            if tree_data.get("truncated"):
                self._log_message("A árvore do repositório é muito grande e foi truncada pela API do GitHub. Analisando arquivos disponíveis.", "warning")

            total_items = len(tree_data.get("tree", []))
            processed_items = 0
            files_content_read_count = 0 # Contador para o novo limite

            items_iterable = tqdm(tree_data.get("tree", []), desc="Processando arquivos", unit="item", disable=progress_callback is not None)

            for item in items_iterable:
                processed_items += 1
                if progress_callback:
                    progress_callback(processed_items / total_items if total_items > 0 else 0, f"Analisando: {item['path']}")

                if item["type"] == "blob": # Arquivo
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
                        # --- ALTERAÇÃO: Lógica de leitura de conteúdo com limite maior ---
                        if files_content_read_count < self.max_files_to_read_content:
                            # Adiciona uma pausa maior para evitar rate limit com mais arquivos
                            time.sleep(0.15)
                            content = self._fetch_file_content_by_sha(item["sha"], file_path)
                            if content:
                                file_info["content"] = content # Limite de chars por arquivo aplicado na leitura
                                files_content_read_count += 1
                                if files_content_read_count % 50 == 0: # Log a cada 50 arquivos
                                    self._log_message(f"Lido conteúdo de {files_content_read_count} arquivos...", "info")

                        # Mesmo se não ler o conteúdo, adiciona à lista para ter o path
                        self.code_files.append(file_info)
                    # Ignorar outros tipos de arquivo para simplificar

            # --- ALTERAÇÃO: Log do limite ---
            if files_content_read_count >= self.max_files_to_read_content:
                 self._log_message(f"Limite de {self.max_files_to_read_content} arquivos com conteúdo lido atingido.", "warning")
            self._log_message(f"Encontrados {len(self.code_files)} arquivos relevantes. Conteúdo lido para {files_content_read_count} arquivos.", "info")
            # -------------------------------

            if not self.readme_content:
                self._log_message("README.md não encontrado ou vazio.", "warning")
            return True

        except requests.RequestException as e:
            status_code = e.response.status_code if hasattr(e, 'response') and e.response is not None else 'N/A'
            self._log_message(f"Erro ao buscar estrutura do repositório (Status: {status_code}): {e}", "error")
            if status_code == 404:
                 self._log_message(f"Verifique se o repo '{self.repo_owner}/{self.repo_name}' e a branch '{self.branch}' existem.", "error")
            elif status_code == 403:
                 self._log_message("Erro 403: Limite de taxa da API GitHub ou token inválido/ausente.", "error")
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
                    self._log_message(f"Erro ao decodificar {file_path}: {decode_err}", "warning")
                    return None # Falha na decodificação

                # --- ALTERAÇÃO: Limite de caracteres por arquivo aumentado ---
                limited_content = decoded_content[:self.max_chars_per_file_read]
                if len(decoded_content) > self.max_chars_per_file_read:
                     self._log_message(f"Conteúdo do arquivo '{file_path}' truncado em {self.max_chars_per_file_read} caracteres.", "warning")
                return limited_content
                # --------------------------------------------------------
            else:
                 self._log_message(f"Arquivo '{file_path}' não está em base64 ou não tem conteúdo na resposta da API.", "warning")
                 return None

        except (requests.RequestException, json.JSONDecodeError) as e:
            self._log_message(f"Erro ao buscar conteúdo do arquivo '{file_path}': {e}", "warning")
            return None
        except Exception as e:
            self._log_message(f"Erro inesperado ao buscar conteúdo do arquivo '{file_path}': {e}", "warning")
            return None

    def analyze_repository(self):
        """Analisa o repositório e prepara dados para a IA."""
        if not self.code_files and not self.readme_content:
            self._log_message("Nenhum arquivo de código ou README para analisar.", "warning")
            return False
        self._log_message("Analisando conteúdo do repositório...", "info")
        self._extract_repo_summary()
        self._analyze_code_structure()
        self._identify_key_components()
        self._log_message("Análise inicial concluída.", "success")
        return True

    def _extract_repo_summary(self):
        """Extrai informações sumárias do README."""
        self.repo_summary["title"] = f"{self.repo_name}"
        self.repo_summary["owner"] = self.repo_owner
        self.repo_summary["url"] = self.repo_url
        self.repo_summary["branch"] = self.branch

        if self.readme_content:
            match_title = re.search(r'^#\s+(.*?)\n', self.readme_content, re.MULTILINE)
            if match_title:
                self.repo_summary["title"] = match_title.group(1).strip()

            content_after_title = re.sub(r'^#\s+.*?\n', '', self.readme_content, count=1)
            content_cleaned = re.sub(r'^\[!\[.*?\]\(.*?\)\]\(.*?\)\s*?\n?', '', content_after_title, flags=re.MULTILINE)
            match_desc = re.search(r'^\s*(.*?)\n\n', content_cleaned, re.DOTALL)

            if match_desc:
                desc = match_desc.group(1).strip()
                desc_cleaned = ' '.join(desc.split())
                # Aumenta um pouco a descrição permitida
                self.repo_summary["description"] = textwrap.shorten(desc_cleaned, width=500, placeholder="...")
            else:
                 fallback_desc = '\n'.join(content_cleaned.strip().splitlines()[:5])
                 self.repo_summary["description"] = textwrap.shorten(fallback_desc, width=500, placeholder="...")
        else:
            self.repo_summary["description"] = f"Um repositório GitHub de {self.repo_owner} sem README.md detalhado."

        self._log_message(f"Título: {self.repo_summary['title']}", "info")
        self._log_message(f"Descrição: {self.repo_summary['description']}", "info")

    def _analyze_code_structure(self):
        """Analisa os arquivos de código e sua estrutura."""
        self.repo_summary["languages"] = {}
        self.repo_summary["file_count"] = len(self.code_files)
        total_code_files_with_content = sum(1 for f in self.code_files if f.get("content"))

        for file in self.code_files:
            if file["extension"]:
                ext = file["extension"][1:]
                self.repo_summary["languages"][ext] = self.repo_summary["languages"].get(ext, 0) + 1

        self.repo_summary["languages"] = dict(sorted(self.repo_summary["languages"].items(), key=lambda item: item[1], reverse=True))
        self._log_message(f"Linguagens detectadas: {self.repo_summary['languages']}", "info")
        self._log_message(f"{total_code_files_with_content} arquivos tiveram conteúdo carregado para análise.", "info")

    def _identify_key_components(self):
        """Identifica componentes chave e arquivos importantes (heurística simples)."""
        key_files = []
        patterns = {
            'config': ['config.', 'settings.', 'env', '.config', 'conf.', 'manifest.', 'docker-compose', 'values.yaml'],
            'entry': ['main.', 'app.', 'index.', 'server.', 'run.', '__main__.', 'wsgi.', 'asgi.'],
            'build': ['package.json', 'requirements.txt', 'pom.xml', 'build.gradle', 'dockerfile', 'makefile', 'setup.py', 'go.mod', 'cargo.toml'],
            'docs': ['contributing.md', 'license', 'code_of_conduct.md', 'changelog.md'],
            'test': ['test_', '_test.', '.spec.', '.test.'],
            'workflow': ['.github/workflows', '.gitlab-ci.yml']
        }
        # Adiciona mais alguns arquivos chave comuns
        common_files = ['manage.py', 'vite.config.', 'webpack.config.', 'next.config.', 'nuxt.config.']

        for file in self.code_files:
            fn_lower = file["name"].lower()
            path_lower = file["path"].lower()
            matched_type = None

            # Verifica padrões por nome
            for type, prefixes in patterns.items():
                for prefix in prefixes:
                    if fn_lower.startswith(prefix) or fn_lower == prefix:
                        matched_type = type
                        break
                if matched_type: break

            # Verifica padrões por caminho (para workflows, testes)
            if not matched_type:
                 for type, prefixes in patterns.items():
                     for prefix in prefixes:
                          if prefix in path_lower: # Verifica se o prefixo está no caminho
                               if type == 'test' and 'test' in path_lower.split('/'): matched_type = type
                               elif type == 'workflow' and '.github/workflows' in path_lower : matched_type = type
                               break
                     if matched_type: break

            # Verifica arquivos comuns exatos
            if not matched_type:
                 if fn_lower in common_files:
                     matched_type = 'entry/config' # Categoria genérica

            if matched_type:
                 key_files.append({"path": file["path"], "type": matched_type})

        # Limita a lista enviada no sumário (mas a IA terá acesso a mais arquivos no contexto)
        self.repo_summary["key_files_guess"] = key_files[:20]
        self._log_message(f"Possíveis arquivos chave identificados: {[f['path'] for f in self.repo_summary['key_files_guess']]}", "info")

    def generate_podcast_script(self, progress_callback=None):
        """Gera o script DETALHADO do podcast usando o modelo Gemini de contexto longo."""
        self._log_message(f"Gerando script DETALHADO do podcast com {self.gemini_model_name}...", "info")

        if not self.gemini_api_key:
            self._log_message("API Key da Gemini não configurada. Gerando script de exemplo.", "error")
            return self._generate_stub_podcast_script()

        # --- ALTERAÇÃO: Preparação de dados para IA com limites maiores ---
        if progress_callback: progress_callback(0.65, "Preparando dados para IA (contexto amplo)...")

        key_file_paths = [kf['path'] for kf in self.repo_summary.get('key_files_guess', [])]
        files_with_content = sorted(
            [f for f in self.code_files if f.get('content')],
            key=lambda f: (0 if f['path'] in key_file_paths else 1, f['path'])
        )

        code_snippets_for_ai = []
        total_code_chars_collected = 0

        self._log_message(f"Tentando incluir até ~{self.max_total_code_chars_for_ai / 1000:.0f}k caracteres de código no prompt para a IA.", "info")

        for file in files_with_content:
            content_to_add = file['content'] # Usa o conteúdo já limitado na leitura (max_chars_per_file_read)
            # Verifica se adicionar este arquivo não estoura o limite TOTAL para código
            if total_code_chars_collected + len(content_to_add) <= self.max_total_code_chars_for_ai:
                code_snippets_for_ai.append({
                    "path": file["path"],
                    # Não precisa truncar aqui, já foi truncado na leitura
                    "content": content_to_add
                })
                total_code_chars_collected += len(content_to_add)
            else:
                # Se estourar o limite TOTAL, para de adicionar arquivos
                self._log_message(f"Limite total de {self.max_total_code_chars_for_ai / 1000:.0f}k caracteres de código para IA atingido. {len(code_snippets_for_ai)}/{len(files_with_content)} arquivos com conteúdo incluídos no prompt.", "warning")
                break

        # Prepara o README, aplicando o limite específico para ele
        readme_for_ai = self.readme_content[:self.max_readme_chars_for_ai]
        if len(self.readme_content) > self.max_readme_chars_for_ai:
             self._log_message(f"Conteúdo do README truncado em {self.max_readme_chars_for_ai} caracteres para a IA.", "warning")

        # Prepara a lista completa de arquivos, aplicando o limite
        all_files_list_for_ai = [f["path"] for f in self.code_files][:self.max_file_list_for_ai]
        if len(self.code_files) > self.max_file_list_for_ai:
             self._log_message(f"Lista de arquivos truncada em {self.max_file_list_for_ai} nomes para a IA.", "warning")


        repo_data_for_ai = {
            "analysis_level": "detailed",
            "summary": self.repo_summary,
            "readme_content": readme_for_ai,
            "code_files_analyzed_count": len(code_snippets_for_ai),
            "code_files_analyzed": code_snippets_for_ai,
            "all_files_list": all_files_list_for_ai
        }
        # -----------------------------------------------------------------

        # --- ALTERAÇÃO: Prompt atualizado para pedir mais detalhes (já definido anteriormente) ---
        prompt = f"""
        Você é um Gerador de Roteiros de Podcast EXTREMAMENTE DETALHISTA chamado 'Código Aberto Explica Profundamente'. Sua missão é criar um roteiro de podcast **técnico, aprofundado e didático** em português do Brasil (pt-BR) explicando um repositório do GitHub para **desenvolvedores que querem entender o código a fundo (nível júnior a pleno)**.

        **Repositório Alvo:** (Informações básicas)
        - URL: {self.repo_url}
        - Proprietário: {self.repo_owner}
        - Nome: {self.repo_name}
        - Branch: {self.branch}

        **Dados Coletados do Repositório (Amplo Contexto Fornecido):**
        Você recebeu um GRANDE volume de informações, incluindo o README e o conteúdo de DIVERSOS arquivos de código. Use este contexto amplo para fornecer uma análise detalhada. O JSON abaixo contém o resumo, o README (limitado a {len(readme_for_ai)} chars), uma lista com os nomes de até {len(all_files_list_for_ai)} arquivos, e o conteúdo detalhado de {len(code_snippets_for_ai)} arquivos de código (totalizando aproximadamente {total_code_chars_collected / 1000:.0f}k caracteres de código).

        **Seu Roteiro de Podcast (Aproximadamente 10-15 minutos de fala - MAIS LONGO E DETALHADO):**

        **Objetivo Principal:** Destrinchar o repositório, aproveitando o extenso contexto de código fornecido. Vá além do superficial.
        1.  **Propósito e Contexto:** Problema resolvido, nicho de aplicação, motivação (inferir do README e código).
        2.  **Arquitetura Detalhada:** Padrões de projeto usados (MVC, Microservices, etc.?), fluxo de dados principal, como os componentes principais (identificados nos arquivos) interagem. **Cite arquivos e trechos de código específicos como exemplo.**
        3.  **Tecnologias e Bibliotecas Chave:** Não apenas liste, mas explique *por que* você acha que foram escolhidas e *como* são usadas no contexto do projeto. Encontrou alguma configuração interessante?
        4.  **Análise de Código Específica (Onde o contexto amplo brilha):**
            *   Identifique 2-3 **algoritmos ou lógicas de negócio complexas/interessantes** presentes no código fornecido. Explique como funcionam.
            *   Discuta a **qualidade do código**: boas práticas observadas (ou ausência delas), tratamento de erros, testes (se visíveis), estilo de código, modularidade. **Use exemplos dos arquivos analisados.**
            *   Quais **conceitos avançados** um dev precisaria dominar para contribuir significativamente? (Ex: Concorrência, Injeção de Dependência, Design Patterns específicos, etc.).
        5.  **Para o Aprendiz:**
            *   Como um dev júnior/pleno pode **navegar e entender** essa base de código de forma eficaz? Sugira pontos de partida.
            *   Quais partes do código são **bons exemplos didáticos** para aprender conceitos específicos?
            *   Sugestões de **pequenas contribuições ou experimentos** que podem ser feitos no código para aprendizado.

        **Formato do Roteiro:**
        *   Use marcadores como [VINHETA DE ABERTURA], [VINHETA DE TRANSIÇÃO], [VINHETA DE ENCERRAMENTO].
        *   Indique o locutor como "LOCUTOR:".
        *   Estruture em seções claras (Introdução, Arquitetura, Tecnologias, Análise de Código, Dicas de Aprendizado, Conclusão).
        *   Linguagem clara, mas **não tenha medo de ser técnico**. Explique termos complexos brevemente.
        *   **Seja analítico e profundo.** Faça conexões entre diferentes partes do código.
        *   **Priorize a precisão técnica** baseada no código fornecido.

        **Comece o roteiro:**
        [VINHETA DE ABERTURA]

        LOCUTOR: Olá, entusiastas do código! Bem-vindos ao Código Aberto Explica Profundamente...
        """
        # -----------------------------------------------------------------

        try:
            if progress_callback: progress_callback(0.8, f"Gerando roteiro DETALHADO com {self.gemini_model_name} (pode levar vários minutos)...")

            # --- ALTERAÇÃO: Garantir que o modelo correto está sendo chamado ---
            model = genai.GenerativeModel(self.gemini_model_name)

            # Calcular tamanho estimado do input total (aproximado)
            prompt_chars = len(prompt)
            data_chars = len(json.dumps(repo_data_for_ai)) # Tamanho real dos dados como string JSON
            total_input_chars_est = prompt_chars + data_chars
            self._log_message(f"Enviando prompt + dados (~{total_input_chars_est / 1000:.0f}k caracteres estimados) para {self.gemini_model_name}...", "info")

            # A API do google.generativeai geralmente aceita uma lista de conteúdos
            # Pode ser necessário ajustar dependendo de como a API lida com JSONs grandes
            # Alternativa 1: Tudo no prompt (pode não ser ideal)
            # response = model.generate_content(prompt + "\n\n```json\n" + json.dumps(repo_data_for_ai, indent=2) + "\n```", ...)
            # Alternativa 2: Usar a estrutura de partes (melhor para multimodal, mas pode funcionar)
            response = model.generate_content(
                 [prompt, json.dumps(repo_data_for_ai)], # Envia como partes separadas
                 generation_config=self.generation_config,
                 safety_settings=self.safety_settings
            )
            # -----------------------------------------------------------------

            # Verifica se a resposta foi bloqueada por segurança ou outros motivos
            if not response.candidates:
                 block_reason = "Desconhecido"
                 safety_ratings_str = "N/A"
                 if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                     block_reason = response.prompt_feedback.block_reason
                     try:
                         safety_ratings_str = str(response.prompt_feedback.safety_ratings)
                     except Exception: pass # Ignora se não conseguir obter
                 self._log_message(f"Geração bloqueada pela API Gemini ({self.gemini_model_name}). Razão: {block_reason}", "error")
                 self._log_message(f"Classificações de segurança do prompt: {safety_ratings_str}", "warning")
                 return self._generate_stub_podcast_script(f"Erro: Conteúdo bloqueado pela política de segurança da IA ({self.gemini_model_name}). Razão: {block_reason}")

            # Extrai o texto da resposta
            script = response.text # Assumindo que a resposta virá em .text

            script += f"""
            \n\n---\n
            [VINHETA DE ENCERRAMENTO]\n
            Roteiro DETALHADO gerado por Código Aberto Explica Profundamente (usando {self.gemini_model_name}).
            Repositório analisado: {self.repo_url}
            """

            self._log_message("Script DETALHADO do podcast gerado com sucesso!", "success")
            if progress_callback: progress_callback(0.9, "Roteiro detalhado gerado!")
            return script.strip()

        except Exception as e:
             # Tentar capturar erros específicos da API do Gemini se disponíveis
             # (Ex: ResourceExhaustedError, InvalidArgumentError etc., dependendo da lib)
             self._log_message(f"Erro ao gerar script com {self.gemini_model_name}: {e}", "error")
             if "429" in str(e) or "ResourceExhaustedError" in str(e): # Exemplo de verificação
                 self._log_message("Erro 429: Limite de taxa da API Gemini atingido. Tente novamente mais tarde.", "error")
             elif "API key not valid" in str(e):
                  self._log_message("Erro: API Key da Gemini inválida.", "error")

             # Log detalhado do erro
             import traceback
             self._log_message(traceback.format_exc(), "error")

             # Retorna um stub que menciona a falha com o modelo grande
             return self._generate_stub_podcast_script(f"Erro na comunicação com o modelo experimental {self.gemini_model_name}: {e}")

    def _generate_stub_podcast_script(self, error_message="API indisponível ou erro na geração"):
        """Gera um script de exemplo quando a API falha."""
        self._log_message(f"Gerando script de exemplo devido a erro: {error_message}", "warning")
        repo_title = self.repo_summary.get("title", self.repo_name)
        main_language = next(iter(self.repo_summary.get("languages", {"código"}).keys()), "código")
        model_name_mention = f" (tentativa com {self.gemini_model_name})" if "experimental" in error_message else ""

        return f"""
        [VINHETA DE ABERTURA]

        LOCUTOR: Olá! Bem-vindos ao Código Aberto Explica! Hoje tivemos um problema técnico ao tentar gerar a análise{model_name_mention}: {error_message}. Vamos dar uma olhada geral com as informações básicas que temos.

        [VINHETA DE TRANSIÇÃO]

        LOCUTOR: O repositório que estávamos explorando é o '{repo_title}', mantido por '{self.repo_owner}'. Ele parece ser escrito principalmente em {main_language}.

        LOCUTOR: A descrição sugere que o objetivo é: {self.repo_summary.get('description', 'Não foi possível carregar a descrição.')}

        LOCUTOR: Como não conseguimos a análise profunda da IA, a dica para hoje é: sempre comece pelo README! Depois, explore a estrutura de pastas e tente identificar os arquivos de configuração e pontos de entrada principais (como 'main.py', 'index.js', etc.).

        LOCUTOR: Recomendamos que você mesmo navegue pelo repositório no link da descrição.

        [VINHETA DE ENCERRAMENTO]
        ---
        Roteiro de exemplo gerado devido a erro.
        Repositório: {self.repo_url}
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
        try:
            vignette_full = AudioSegment.from_mp3(vignette_path)
            vignette = vignette_full[:vignette_duration_ms]
            vignette = vignette.fade_out(duration=min(500, vignette_duration_ms // 4))
            self._log_message(f"Vinheta '{vignette_path}' carregada e cortada para {len(vignette)/1000:.1f}s.", "info")
        except FileNotFoundError:
            self._log_message(f"Arquivo de vinheta '{vignette_path}' não encontrado. Não será possível adicionar vinhetas.", "error")
            return None
        except Exception as e:
            self._log_message(f"Erro ao carregar ou processar vinheta '{vignette_path}': {e}. Verifique o arquivo e a instalação do ffmpeg.", "error")
            return None

        # --- Processar o roteiro e gerar áudio ---
        final_segments = []
        vignette_markers = [
            '[VINHETA DE ABERTURA]',
            '[MÚSICA SUAVE DE FUNDO]',
            '[SECTION BREAK]',
            '[VINHETA DE TRANSIÇÃO]',
            '[VINHETA DE ENCERRAMENTO]'
        ]
        marker_pattern = r'(' + '|'.join(re.escape(m) for m in vignette_markers) + r')'
        parts = [p.strip() for p in re.split(marker_pattern, script_text) if p and p.strip()]

        if not parts:
             self._log_message("Roteiro vazio ou sem partes reconhecíveis após divisão.", "error")
             return None

        total_parts = len(parts)
        processed_parts = 0

        try:
            for part in parts:
                processed_parts += 1
                current_progress = 0.95 + (0.04 * (processed_parts / total_parts))
                if progress_callback:
                     progress_callback(current_progress, f"Processando áudio parte {processed_parts}/{total_parts}...")

                is_marker = part in vignette_markers

                if is_marker:
                    self._log_message(f"Adicionando vinheta para o marcador: {part}", "info")
                    final_segments.append(vignette)
                else: # É um bloco de texto
                    # 1. Limpeza inicial (metadados, linhas vazias)
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

                    # Limpa espaços extras que podem ter sido deixados pelas substituições
                    text_to_speak = re.sub(r'\s+', ' ', text_to_speak).strip()

                    if text_to_speak:
                        self._log_message(f"Gerando fala para: '{textwrap.shorten(text_to_speak, 50)}...'", "info")
                        try:
                            # --- Geração de Fala ---
                            tts = gTTS(text=text_to_speak, lang=lang, slow=False)
                            speech_fp = io.BytesIO()
                            tts.write_to_fp(speech_fp)
                            speech_fp.seek(0)

                            # --- Carregamento com Pydub ---
                            try:
                                speech_segment = AudioSegment.from_mp3(speech_fp)
                                final_segments.append(speech_segment)
                                self._log_message(f"Segmento de fala gerado ({len(speech_segment)/1000:.1f}s).", "info")
                            except Exception as e_pydub:
                                 self._log_message(f"Erro ao carregar segmento de fala com pydub: {e_pydub}. Verifique ffmpeg. Pulando segmento.", "warning")
                                 continue

                        except Exception as e_tts:
                            self._log_message(f"Erro ao gerar fala para um segmento com gTTS: {e_tts}", "warning")
                            continue

            # --- Combinar todos os segmentos ---
            if not final_segments:
                self._log_message("Nenhum segmento de áudio foi gerado.", "error")
                return None

            self._log_message("Combinando segmentos de fala e vinhetas...", "info")
            combined_audio = AudioSegment.empty()
            for segment in final_segments:
                 combined_audio += segment

            # --- Exportar o áudio final ---
            final_audio_fp = io.BytesIO()
            combined_audio.export(final_audio_fp, format="mp3")
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
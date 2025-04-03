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

# Carrega variáveis de ambiente do .env (se existir)
load_dotenv()

class GitHubPodcastGenerator:
    """
    Ferramenta para gerar podcasts didáticos em áudio explicando repositórios do GitHub,
    focando em desenvolvedores juniores, usando a API Gemini e gTTS.
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
            # Poderíamos lançar um erro aqui, mas vamos permitir continuar para a interface mostrar a mensagem
            # raise ValueError("API Key da Gemini não encontrada. Defina a variável de ambiente GOOGLE_API_KEY ou passe via parâmetro.")
        else:
             try:
                genai.configure(api_key=self.gemini_api_key)
                print("✓ API Gemini configurada.")
             except Exception as e:
                 print(f"Erro ao configurar a API Gemini: {e}")
                 self.gemini_api_key = None # Invalida a chave se a configuração falhar


        self.headers = {}
        if self.github_token:
            self.headers = {"Authorization": f"token {self.github_token}"}
            print("✓ Token do GitHub encontrado.")
        else:
            print("⚠️ Token do GitHub não encontrado. Repositórios privados ou com alto tráfego podem falhar.")

        # Detalhes do repositório
        self.repo_owner = None
        self.repo_name = None
        self.branch = "main" # Default branch
        self.repo_url = None

        # Extensões de código a analisar (pode ser customizado)
        self.code_extensions = ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.h', '.go', '.rb', '.php', '.ts', '.jsx', '.tsx', '.ipynb', '.sh', '.md'] # Adicionado .md para README

        # Armazena dados do repositório
        self.code_files = []
        self.readme_content = ""
        self.repo_summary = {}
        self.generation_config = genai.types.GenerationConfig(
             # Only one candidate for now.
            candidate_count=1,
            #stop_sequences=['x'],
            #max_output_tokens=2048,
            temperature=0.7) # Um pouco de criatividade

        # Configurações de segurança (ajuste conforme necessário)
        self.safety_settings={
            # genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            # genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            # genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            # genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
             # Relaxando um pouco para código, mas pode precisar ajustar
             genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
             genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
             genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
             genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
        }

        # Modelo Gemini a ser usado
        self.gemini_model_name = 'gemini-1.5-flash' # Ou 'gemini-pro' ou outro modelo disponível


    def _log_message(self, message, level="info"):
        """Helper para logar mensagens (pode ser substituído por logging real)."""
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
            self.repo_name = path_parts[1].replace('.git', '') # Remove .git se presente
            self.repo_url = f"https://github.com/{self.repo_owner}/{self.repo_name}" # URL canônica

            # Verifica se uma branch específica foi especificada (formato /tree/branch_name)
            if len(path_parts) > 3 and path_parts[2] == "tree":
                self.branch = path_parts[3]
            else:
                 # Tenta obter a branch padrão da API se não especificada
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
            default_branch = repo_info.get("default_branch", "main") # Usa 'main' como fallback
            self._log_message(f"Branch padrão detectada: {default_branch}", "info")
            return default_branch
        except requests.RequestException as e:
            self._log_message(f"Não foi possível obter a branch padrão (usando 'main'): {e}", "warning")
            return "main" # Retorna 'main' como fallback em caso de erro

    def fetch_repo_structure(self, progress_callback=None):
        """Busca a estrutura do repositório e arquivos importantes."""
        self._log_message("Buscando estrutura do repositório...", "info")
        self.code_files = []
        self.readme_content = ""

        # URL base da API para o conteúdo do repositório na branch correta
        # Usa a API de Git Trees para buscar recursivamente (mais eficiente)
        api_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/git/trees/{self.branch}?recursive=1"

        try:
            response = requests.get(api_url, headers=self.headers)
            response.raise_for_status()
            tree_data = response.json()

            if tree_data.get("truncated"):
                self._log_message("A árvore do repositório é muito grande e foi truncada. Analisando arquivos disponíveis.", "warning")

            total_items = len(tree_data.get("tree", []))
            processed_items = 0

            # Usar tqdm se não houver callback de progresso (para CLI)
            items_iterable = tqdm(tree_data.get("tree", []), desc="Processando arquivos", unit="item", disable=progress_callback is not None)

            for item in items_iterable:
                 processed_items += 1
                 if progress_callback:
                     progress_callback(processed_items / total_items if total_items > 0 else 0, f"Analisando: {item['path']}")

                 if item["type"] == "blob": # 'blob' significa arquivo
                    file_path = item["path"]
                    file_name = os.path.basename(file_path)
                    _, ext = os.path.splitext(file_name)

                    # Processar README
                    if file_name.lower() == "readme.md":
                        content = self._fetch_file_content_by_sha(item["sha"], file_path)
                        if content:
                            self.readme_content = content
                            self._log_message(f"README.md encontrado e processado.", "info")

                    # Processar arquivos de código
                    elif ext.lower() in self.code_extensions:
                        # Otimização: buscar conteúdo apenas se necessário (pode ser feito depois pela IA)
                        # Por agora, vamos buscar alguns para análise inicial
                        # Poderíamos adicionar lógica para priorizar arquivos menores ou em diretórios raiz
                        if len(self.code_files) < 50: # Limitar o número de arquivos para buscar conteúdo agora
                            content = self._fetch_file_content_by_sha(item["sha"], file_path)
                            if content:
                                self.code_files.append({
                                    "path": file_path,
                                    "name": file_name,
                                    "extension": ext.lower(),
                                    "content": content # Guardar conteúdo para análise posterior
                                })
                        else:
                             # Apenas listar arquivos se já temos muitos com conteúdo
                              self.code_files.append({
                                    "path": file_path,
                                    "name": file_name,
                                    "extension": ext.lower(),
                                    "content": None # Indicar que o conteúdo não foi buscado
                                })


            self._log_message(f"Encontrados {len(self.code_files)} arquivos de código relevantes.", "info")
            if not self.readme_content:
                self._log_message("README.md não encontrado ou vazio.", "warning")
            return True

        except requests.RequestException as e:
            status_code = e.response.status_code if hasattr(e, 'response') and e.response is not None else 'N/A'
            self._log_message(f"Erro ao buscar estrutura do repositório (Status: {status_code}): {e}", "error")
            if status_code == 404:
                 self._log_message(f"Verifique se o repositório '{self.repo_owner}/{self.repo_name}' e a branch '{self.branch}' existem.", "error")
            elif status_code == 403:
                 self._log_message("Erro 403: Limite de taxa da API do GitHub atingido ou token inválido/ausente para repositório privado.", "error")
            return False
        except Exception as e:
             self._log_message(f"Erro inesperado ao processar estrutura: {e}", "error")
             return False

    def _fetch_file_content_by_sha(self, sha, file_path):
        """Busca e decodifica o conteúdo do arquivo usando o SHA do blob."""
        api_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/git/blobs/{sha}"
        try:
            # Pequena pausa para evitar rate limiting agressivo
            time.sleep(0.1)
            response = requests.get(api_url, headers=self.headers)
            response.raise_for_status()

            content_data = response.json()
            if content_data.get("encoding") == "base64" and content_data.get("content"):
                # Limitar tamanho do conteúdo decodificado para evitar problemas de memória/API
                decoded_content = base64.b64decode(content_data["content"]).decode('utf-8', errors='replace')
                return decoded_content[:15000] # Limite de caracteres por arquivo
            return None # Retorna None se não for base64 ou não tiver conteúdo

        except (requests.RequestException, json.JSONDecodeError, UnicodeDecodeError) as e:
            self._log_message(f"Erro ao buscar conteúdo do arquivo '{file_path}': {e}", "warning")
            return None
        except Exception as e:
            self._log_message(f"Erro inesperado ao buscar conteúdo do arquivo '{file_path}': {e}", "warning")
            return None

    # Métodos _fetch_file_content e _fetch_directory_content não são mais necessários com a API de Trees

    def analyze_repository(self):
        """Analisa o repositório e prepara dados para a IA."""
        if not self.code_files and not self.readme_content:
            self._log_message("Nenhum arquivo de código ou README para analisar.", "warning")
            return False

        self._log_message("Analisando conteúdo do repositório...", "info")

        # 1. Extrair sumário do repositório a partir do README
        self._extract_repo_summary()

        # 2. Analisar estrutura de código e padrões (contagem de linguagens)
        self._analyze_code_structure()

        # 3. Identificar componentes chave e features (arquivos principais)
        #    Esta parte pode ser melhorada pela IA, mas damos uma pista.
        self._identify_key_components()

        self._log_message("Análise inicial concluída.", "success")
        return True

    def _extract_repo_summary(self):
        """Extrai informações sumárias do README."""
        self.repo_summary["title"] = f"{self.repo_name}" # Fallback title
        self.repo_summary["owner"] = self.repo_owner
        self.repo_summary["url"] = self.repo_url
        self.repo_summary["branch"] = self.branch

        if self.readme_content:
            # Tenta extrair título H1 e primeira descrição significativa
            match_title = re.search(r'^#\s+(.*?)\n', self.readme_content, re.MULTILINE)
            if match_title:
                self.repo_summary["title"] = match_title.group(1).strip()

            # Tenta encontrar o primeiro parágrafo após o título (ou do início)
            # Ignora badges (links que são imagens) comuns no início
            content_after_title = re.sub(r'^#\s+.*?\n', '', self.readme_content, count=1) # Remove linha do título H1
            content_cleaned = re.sub(r'^\[!\[.*?\]\(.*?\)\]\(.*?\)\s*?\n?', '', content_after_title, flags=re.MULTILINE) # Remove badges comuns
            match_desc = re.search(r'^\s*(.*?)\n\n', content_cleaned, re.DOTALL) # Pega até o primeiro parágrafo duplo

            if match_desc:
                desc = match_desc.group(1).strip()
                # Limpa múltiplos espaços/newlines e limita tamanho
                desc_cleaned = ' '.join(desc.split())
                self.repo_summary["description"] = textwrap.shorten(desc_cleaned, width=300, placeholder="...")
            else:
                 # Fallback: pega as primeiras linhas do README limpo
                 fallback_desc = '\n'.join(content_cleaned.strip().splitlines()[:3])
                 self.repo_summary["description"] = textwrap.shorten(fallback_desc, width=300, placeholder="...")
        else:
            self.repo_summary["description"] = f"Um repositório GitHub de {self.repo_owner} sem README.md detalhado."

        self._log_message(f"Título: {self.repo_summary['title']}", "info")
        self._log_message(f"Descrição: {self.repo_summary['description']}", "info")


    def _analyze_code_structure(self):
        """Analisa os arquivos de código e sua estrutura."""
        self.repo_summary["languages"] = {}
        self.repo_summary["file_count"] = len(self.code_files)
        total_code_files_with_content = 0

        for file in self.code_files:
            # Conta apenas se tiver extensão (ignora arquivos como 'LICENSE')
            if file["extension"]:
                ext = file["extension"][1:]  # Remove o ponto
                self.repo_summary["languages"][ext] = self.repo_summary["languages"].get(ext, 0) + 1
                if file["content"] is not None:
                    total_code_files_with_content +=1

        # Ordena linguagens por frequência
        self.repo_summary["languages"] = dict(sorted(self.repo_summary["languages"].items(), key=lambda item: item[1], reverse=True))
        self._log_message(f"Linguagens detectadas: {self.repo_summary['languages']}", "info")
        self._log_message(f"{total_code_files_with_content} arquivos tiveram conteúdo carregado para análise.", "info")


    def _identify_key_components(self):
        """Identifica componentes chave e arquivos importantes."""
        # Esta função é simplista. A IA fará um trabalho melhor.
        # Apenas identificamos arquivos comuns de entrada/configuração.
        key_files = []
        patterns = {
            'config': ['config.', 'settings.', 'env', '.config', 'conf.', 'manifest.'],
            'entry': ['main.', 'app.', 'index.', 'server.', 'run.', '__main__.'],
            'build': ['package.json', 'requirements.txt', 'pom.xml', 'build.gradle', 'dockerfile', 'makefile', 'setup.py'],
            'docs': ['contributing.md', 'license', 'code_of_conduct.md']
        }

        for file in self.code_files:
            fn_lower = file["name"].lower()
            matched = False
            for type, prefixes in patterns.items():
                for prefix in prefixes:
                    if fn_lower.startswith(prefix) or fn_lower == prefix:
                        key_files.append({"path": file["path"], "type": type})
                        matched = True
                        break # Próximo arquivo
                if matched:
                    break

        self.repo_summary["key_files_guess"] = key_files[:10] # Limita a lista
        self._log_message(f"Possíveis arquivos chave identificados: {[f['path'] for f in self.repo_summary['key_files_guess']]}", "info")

    def generate_podcast_script(self, progress_callback=None):
        """Gera o script do podcast usando a API Gemini."""
        self._log_message("Gerando script do podcast com Gemini...", "info")

        if not self.gemini_api_key:
            self._log_message("API Key da Gemini não configurada. Gerando script de exemplo.", "error")
            return self._generate_stub_podcast_script()

        # Preparar dados para a IA Gemini
        # Inclui resumo, README (limitado) e trechos de código (limitados)
        # Foca nos arquivos com conteúdo carregado
        code_snippets = []
        chars_so_far = 0
        max_chars_code = 15000 # Limite de caracteres para trechos de código no prompt

        files_with_content = [f for f in self.code_files if f.get('content')]
        # Priorizar arquivos chave, se identificados
        key_file_paths = [kf['path'] for kf in self.repo_summary.get('key_files_guess', [])]
        sorted_files = sorted(files_with_content, key=lambda f: 0 if f['path'] in key_file_paths else 1)


        for file in sorted_files:
            content_snippet = file['content'][:1000] # Pega os primeiros 1000 chars de cada arquivo relevante
            if chars_so_far + len(content_snippet) < max_chars_code:
                code_snippets.append({
                    "path": file["path"],
                    "content_snippet": content_snippet
                })
                chars_so_far += len(content_snippet)
            else:
                self._log_message(f"Limite de caracteres para trechos de código atingido ({max_chars_code}). Nem todos os arquivos incluídos no prompt.", "warning")
                break # Para de adicionar trechos

        repo_data_for_ai = {
            "summary": self.repo_summary,
            "readme_preview": self.readme_content[:3000] if self.readme_content else "README não disponível ou vazio.", # Limita tamanho do README
            "code_snippets": code_snippets,
            "all_files_list": [f["path"] for f in self.code_files][:100] # Lista de todos os arquivos (limitada)
        }

        # Prompt Detalhado para Gemini
        prompt = f"""
        Você é um Gerador de Roteiros de Podcast chamado 'Código Aberto Explica'. Sua missão é criar um roteiro de podcast **didático e engajador** em português do Brasil (pt-BR) explicando um repositório do GitHub para **desenvolvedores júnior**.

        **Repositório Alvo:**
        - URL: {self.repo_url}
        - Proprietário: {self.repo_owner}
        - Nome: {self.repo_name}
        - Branch: {self.branch}

        **Dados Coletados do Repositório (Resumo e Amostras):**
        ```json
        {json.dumps(repo_data_for_ai, indent=2, ensure_ascii=False)}
        ```

        **Seu Roteiro de Podcast (Aproximadamente 5-7 minutos de fala):**

        **Objetivo Principal:** Explicar o repositório de forma clara para um desenvolvedor júnior, focando em:
        1.  **Propósito:** Qual problema o projeto resolve? Qual seu objetivo principal? (Use o README e o contexto).
        2.  **Estrutura e Arquitetura:** Como o código está organizado? Quais são as pastas e arquivos principais? Qual a lógica geral? (Inferir da lista de arquivos e trechos).
        3.  **Tecnologias Chave:** Quais linguagens, frameworks ou bibliotecas importantes são usadas? (Baseado nas extensões e trechos).
        4.  **Para o Júnior:**
            *   Quais **conceitos de programação** ou **tecnologias** um júnior **precisa entender** para começar a trabalhar com um código como este? (Ex: APIs REST, ORM, estado em React, etc.). Mencione 2-3 conceitos chave.
            *   Como um júnior poderia **começar a construir algo similar**? Quais seriam os primeiros passos? (Ex: "Comece com um servidor web simples...", "Crie a estrutura básica de pastas...", "Defina os modelos de dados...").
            *   Há alguma **boa prática** evidente no código (mesmo nas amostras) que vale a pena destacar? (Ex: Nomes de variáveis claros, comentários úteis, estrutura modular).
        5.  **Como Aprender Mais:** Sugira como o ouvinte pode usar ou aprender com este repositório (Ex: "Clone o repo e tente rodar", "Leia a documentação", "Contribua com uma issue pequena").

        **Formato do Roteiro:**
        *   Use [VINHETA DE ABERTURA] e [VINHETA DE ENCERRAMENTO].
        *   Use [MÚSICA SUAVE DE FUNDO] para transições ou ênfase.
        *   Indique o locutor como "LOCUTOR:".
        *   Divida o conteúdo em seções lógicas (Introdução, Estrutura, Tecnologias, Dicas para Juniores, Conclusão).
        *   A linguagem deve ser **acessível, clara e motivadora**, evitando jargões excessivos ou explicando-os brevemente.
        *   **Seja criativo e didático!** Imagine que você está conversando com um colega júnior.
        *   **Não invente detalhes não presentes nos dados.** Se a informação não está clara, admita ("Parece que...", "Pela estrutura, podemos inferir...").
        *   **Foco no aspecto educacional para juniores.**

        **Comece o roteiro agora:**
        [VINHETA DE ABERTURA]

        LOCUTOR: Olá, desenvolvedor e desenvolvedora! Bem-vindos ao Código Aberto Explica...
        """

        try:
            if progress_callback:
                 progress_callback(0.8, "Gerando roteiro com a IA...") # Atualiza progresso antes da chamada

            model = genai.GenerativeModel(self.gemini_model_name)
            response = model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
                )

            # Verifica se a resposta foi bloqueada por segurança
            if not response.candidates:
                 block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else 'Desconhecido'
                 self._log_message(f"Geração bloqueada pela API Gemini. Razão: {block_reason}", "error")
                 # Tentar obter informações sobre a classificação de segurança, se disponíveis
                 try:
                     safety_ratings = response.prompt_feedback.safety_ratings
                     self._log_message(f"Classificações de segurança: {safety_ratings}", "warning")
                 except Exception:
                     pass # Ignora se não conseguir obter detalhes
                 return self._generate_stub_podcast_script("Erro: Conteúdo bloqueado pela política de segurança da IA.")


            # Extrai o texto da resposta
            script = response.text

            # Adiciona metadados ao final
            script += f"""
            \n\n---\n
            [VINHETA DE ENCERRAMENTO]\n
            Roteiro gerado por Código Aberto Explica (usando Google Gemini).
            Repositório analisado: {self.repo_url}
            """

            self._log_message("Script do podcast gerado com sucesso!", "success")
            if progress_callback:
                 progress_callback(0.9, "Roteiro gerado!")
            return script.strip()

        except Exception as e:
            self._log_message(f"Erro ao gerar script com Gemini: {e}", "error")
            # Tenta fornecer mais detalhes se for um erro da API
            if hasattr(e, 'response'):
                 self._log_message(f"Detalhes do erro da API: {e.response.text}", "error")
            return self._generate_stub_podcast_script(f"Erro na comunicação com a API Gemini: {e}")

    def _generate_stub_podcast_script(self, error_message="API indisponível"):
        """Gera um script de exemplo quando a API falha."""
        self._log_message(f"Gerando script de exemplo devido a erro: {error_message}", "warning")
        repo_title = self.repo_summary.get("title", self.repo_name)
        main_language = next(iter(self.repo_summary.get("languages", {"código"}).keys()), "código")

        return f"""
        [VINHETA DE ABERTURA]

        LOCUTOR: Olá! Bem-vindos ao Código Aberto Explica! Hoje tivemos um pequeno problema técnico ({error_message}) para analisar profundamente o repositório {repo_title}, mas vamos dar uma olhada geral baseada nas informações que conseguimos!

        [MÚSICA SUAVE DE FUNDO]

        LOCUTOR: O repositório que íamos explorar hoje é o '{repo_title}', mantido por '{self.repo_owner}'. Pelo que vimos, parece ser um projeto escrito principalmente em {main_language}.

        LOCUTOR: A descrição sugere que o objetivo é: {self.repo_summary.get('description', 'Não foi possível carregar a descrição.')}

        LOCUTOR: Como não conseguimos a análise completa da IA, a dica para o júnior hoje é: sempre comece explorando o README! Ele é o cartão de visitas do projeto. Depois, olhe a estrutura de pastas e tente rodar o projeto localmente, se possível.

        LOCUTOR: Recomendamos que você mesmo dê uma olhada no link do repositório na descrição deste episódio!

        [VINHETA DE ENCERRAMENTO]
        ---
        Roteiro de exemplo gerado devido a erro.
        Repositório: {self.repo_url}
        """

    def generate_podcast_audio(self, script_text, lang='pt-br', progress_callback=None):
        """Gera o áudio do podcast a partir do script usando gTTS."""
        self._log_message("Gerando áudio do podcast...", "info")
        if progress_callback:
            progress_callback(0.95, "Sintetizando áudio...")

        try:
            # Remove marcadores como [VINHETA...] para não serem lidos
            text_to_speak = re.sub(r'\[.*?\]', '', script_text)
            # Remove linhas de metadados no final
            text_to_speak = re.sub(r'---\n.*', '', text_to_speak, flags=re.DOTALL).strip()

            tts = gTTS(text=text_to_speak, lang=lang, slow=False)

            # Salva o áudio em um buffer de bytes na memória
            audio_fp = io.BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0) # Volta para o início do buffer para leitura

            self._log_message("Áudio gerado com sucesso!", "success")
            if progress_callback:
                 progress_callback(1.0, "Podcast pronto!")
            return audio_fp

        except Exception as e:
            self._log_message(f"Erro ao gerar áudio com gTTS: {e}", "error")
            if "403 (Forbidden)" in str(e):
                 self._log_message("Erro 403 no gTTS: Possível excesso de requisições. Tente novamente mais tarde.", "error")
            return None
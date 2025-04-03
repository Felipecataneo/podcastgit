import streamlit as st
import os
from github_podcast_generator import GitHubPodcastGenerator # Importa a classe refatorada
from dotenv import load_dotenv
import re

# Carrega variáveis de ambiente (útil para desenvolvimento local com .env)
load_dotenv()

# --- Configuração da Página ---
st.set_page_config(
    page_title="Podcast Explica Código",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Funções Auxiliares ---
def is_valid_github_url(url):
    """Verifica se a URL parece ser uma URL válida do GitHub."""
    pattern = r'^https://github\.com/[\w\-]+/[\w\-\.]+(?:/tree/[\w\-\.]+)?/?$'
    return re.match(pattern, url) is not None

# --- Cache para a Classe (Opcional, mas pode ajudar a reter estado entre reruns) ---
# @st.cache_resource # Desativado por enquanto, pode causar problemas com estado interno
def get_generator(api_key, github_token):
     # Garante que temos uma chave válida antes de instanciar
     if not api_key:
         st.error("API Key da Gemini é necessária para continuar.")
         st.stop() # Impede a execução do restante do script se não houver chave

     try:
        # Passa os tokens explicitamente
        return GitHubPodcastGenerator(gemini_api_key=api_key, github_token=github_token)
     except Exception as e:
         st.error(f"Erro ao inicializar o gerador: {e}")
         st.stop()


# --- Interface do Usuário (Sidebar) ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/25/25231.png", width=80) # Logo GitHub
st.sidebar.title("🎙️ Podcast Explica Código")
st.sidebar.markdown("Transforme um repositório GitHub em um podcast didático para devs juniores!")

st.sidebar.header("1. Configuração")

# Obter API Key (Prioriza secrets, depois .env, depois input)
gemini_api_key_input = os.getenv("GOOGLE_API_KEY") # Tenta pegar do .env primeiro
if not gemini_api_key_input and 'GOOGLE_API_KEY' in st.secrets:
     gemini_api_key_input = st.secrets["GOOGLE_API_KEY"] # Tenta pegar dos secrets do Streamlit

gemini_api_key = st.sidebar.text_input(
    "Sua API Key do Google Gemini",
    type="password",
    value=gemini_api_key_input or "", # Preenche se encontrou
    help="Necessária para gerar o roteiro. Obtenha em [Google AI Studio](https://aistudio.google.com/app/apikey)"
)

# Obter GitHub Token (Opcional, mas recomendado)
github_token_input = os.getenv("GITHUB_TOKEN")
if not github_token_input and 'GITHUB_TOKEN' in st.secrets:
     github_token_input = st.secrets["GITHUB_TOKEN"]

github_token = st.sidebar.text_input(
    "Seu GitHub Token (Opcional)",
    type="password",
    value=github_token_input or "",
    help="Recomendado para evitar limites de taxa e acessar repositórios privados. [Crie um aqui](https://github.com/settings/tokens)"
)


st.sidebar.header("2. Repositório")
repo_url = st.sidebar.text_input("URL do Repositório GitHub", placeholder="Ex: https://github.com/usuario/projeto")

# --- Lógica Principal ---
st.header("Resultado da Geração")

if st.sidebar.button("Gerar Podcast!", use_container_width=True, type="primary"):
    if not gemini_api_key:
        st.warning("Por favor, insira sua API Key da Google Gemini na barra lateral.")
    elif not repo_url:
        st.warning("Por favor, insira a URL do repositório GitHub na barra lateral.")
    elif not is_valid_github_url(repo_url):
         st.error("URL do GitHub inválida. Use o formato: https://github.com/usuario/repositorio")
    else:
        # Inicializa o gerador (pode mostrar erro e parar aqui se a chave for inválida na inicialização)
        try:
            generator = get_generator(gemini_api_key, github_token)
        except Exception as e: # Captura erro se get_generator falhar e parar
             # A mensagem de erro já foi mostrada por get_generator
             st.stop()


        # Placeholder para o progresso e logs
        progress_bar = st.progress(0, text="Iniciando...")
        log_area = st.expander("Logs Detalhados", expanded=False)
        log_messages = []

        def update_progress(percentage, message):
            """Função de callback para atualizar a barra de progresso e logs."""
            progress_bar.progress(percentage, text=message)
            log_messages.append(message)
            # Atualiza a área de logs dinamicamente (pode causar re-render completo)
            # log_area.text("\n".join(log_messages)) # Comentado pois pode ser lento
            print(f"Progresso: {int(percentage*100)}% - {message}") # Log no console também

        try:
            with st.spinner("Analisando URL e buscando repositório..."):
                 if not generator.parse_github_url(repo_url):
                     st.error("Não foi possível analisar a URL do GitHub. Verifique a URL e tente novamente.")
                     st.stop()

                 update_progress(0.1, f"Analisando {generator.repo_owner}/{generator.repo_name}...")

                 if not generator.fetch_repo_structure(progress_callback=update_progress):
                     st.error("Falha ao buscar a estrutura do repositório. Verifique a URL, a branch, o token do GitHub (se privado) e sua conexão.")
                     st.stop()

                 update_progress(0.5, "Estrutura do repositório obtida.")

                 if not generator.analyze_repository():
                      st.warning("Análise inicial do repositório concluída, mas pode haver poucos dados.")
                      # Não paramos aqui, tentamos gerar mesmo assim
                 update_progress(0.6, "Análise inicial concluída.")


            with st.spinner("Gerando roteiro do podcast com a IA... (Pode levar um tempo)"):
                 podcast_script = generator.generate_podcast_script(progress_callback=update_progress)
                 if not podcast_script or "Erro:" in podcast_script:
                      st.error(f"Falha ao gerar o roteiro do podcast. Verifique os logs.")
                      log_area.error(podcast_script) # Mostra erro no log
                      st.stop()
                 update_progress(0.9, "Roteiro gerado!")

            with st.spinner("Gerando áudio do podcast..."):
                 podcast_audio_bytes = generator.generate_podcast_audio(podcast_script, progress_callback=update_progress)
                 if not podcast_audio_bytes:
                     st.error("Falha ao gerar o áudio do podcast. Verifique os logs e tente novamente.")
                     log_area.text("\n".join(log_messages)) # Mostra logs acumulados
                     st.stop()

            # --- Exibe os Resultados ---
            st.success("Podcast gerado com sucesso! 🎉")

            # Nome do arquivo para download
            safe_repo_name = re.sub(r'[^\w\-]+', '_', generator.repo_name)
            audio_filename = f"{safe_repo_name}_podcast.mp3"
            script_filename = f"{safe_repo_name}_roteiro.md"

            # Player de Áudio
            st.subheader("🎧 Ouça o Podcast:")
            st.audio(podcast_audio_bytes, format="audio/mp3")

            # Botão de Download do Áudio
            st.download_button(
                label="⬇️ Baixar Áudio (MP3)",
                data=podcast_audio_bytes,
                file_name=audio_filename,
                mime="audio/mp3",
                use_container_width=True
            )

            # Exibir e permitir download do roteiro
            st.subheader("📜 Roteiro Gerado:")
            st.markdown(f"```markdown\n{podcast_script}\n```") # Usar markdown para melhor formatação
            # st.text_area("Roteiro:", podcast_script, height=400) # Alternativa com text_area

            st.download_button(
                label="⬇️ Baixar Roteiro (MD)",
                data=podcast_script.encode('utf-8'), # Codifica para bytes
                file_name=script_filename,
                mime="text/markdown",
                use_container_width=True
            )

            # Atualiza logs finais
            log_area.text("\n".join(log_messages))
            log_area.success("Processo concluído com sucesso.")


        except Exception as e:
            st.error(f"Ocorreu um erro inesperado durante o processo: {e}")
            log_messages.append(f"ERRO INESPERADO: {e}")
            log_area.text("\n".join(log_messages)) # Mostra logs mesmo em erro
            import traceback
            log_area.text(traceback.format_exc()) # Log completo do erro


# --- Rodapé ou Informações Adicionais ---
st.markdown("---")
st.markdown("Desenvolvido com Streamlit, Google Gemini e gTTS.")
st.markdown("Repositório no GitHub: [link-para-seu-repo-se-quiser]") # Adicione o link do seu projeto aqui
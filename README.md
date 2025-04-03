# 🎙️ Podcast Explica Código

Transforme repositórios do GitHub em podcasts didáticos! Esta ferramenta utiliza a API Google Gemini para analisar o código-fonte e gerar um roteiro explicativo, focado em desenvolvedores juniores a plenos. Em seguida, converte o roteiro em um arquivo de áudio MP3 usando gTTS, com vinhetas musicais adicionadas via Pydub para uma experiência mais agradável.


## ✨ Funcionalidades Principais

*   **Análise Inteligente com IA:** Utiliza a API Google Gemini (potencialmente modelos de contexto amplo) para analisar a estrutura, o código e o propósito de um repositório GitHub.
*   **Roteiro Didático:** Gera automaticamente um script de podcast em português (pt-BR), explicando o repositório de forma clara e estruturada. (O nível de detalhe pode ser ajustado via prompt).
*   **Geração de Áudio (Podcast):** Converte o roteiro gerado em um arquivo de áudio MP3 usando Google Text-to-Speech (gTTS).
*   **Vinhetas Musicais:** Adiciona automaticamente uma vinheta musical (arquivo `background_music.mp3` por padrão) no início, transições e fim do podcast usando Pydub.
*   **Interface Web Simples:** Interface criada com Streamlit para facilitar a inserção da URL do repositório, chaves de API e a geração/download dos resultados.
*   **Suporte a Repositórios Públicos e Privados:** Utiliza um Token do GitHub (opcional, mas recomendado) para acessar repositórios privados e evitar limites de taxa da API.
*   **Feedback de Progresso:** Exibe uma barra de progresso e logs detalhados durante o processo de análise e geração.

## 🚀 Como Funciona

1.  **Input:** O usuário fornece a URL de um repositório GitHub na interface do Streamlit.
2.  **Análise da URL:** O código extrai o proprietário, nome do repositório e branch.
3.  **Busca de Dados:** Utiliza a API do GitHub para buscar a estrutura de arquivos e o conteúdo do README e de arquivos de código relevantes (respeitando limites configuráveis).
4.  **Análise com IA (Gemini):** Envia as informações coletadas (resumo, README, código) para a API Google Gemini com um prompt específico para gerar um roteiro de podcast didático.
5.  **Geração de Roteiro:** A IA retorna o roteiro em formato de texto/markdown.
6.  **Geração de Áudio (gTTS):** O roteiro é processado, limpando marcadores e formatação, e cada segmento de fala é convertido em áudio.
7.  **Adição de Vinhetas (Pydub):** Segmentos de fala e vinhetas musicais são combinados em um único arquivo de áudio.
8.  **Output:** A interface exibe o roteiro gerado e um player para o áudio, com botões para download de ambos (roteiro `.md`, áudio `.mp3`).

## 🛠️ Tecnologias Utilizadas

*   **Python 3.x**
*   **Streamlit:** Para a interface web interativa.
*   **Google Generative AI SDK (`google-generativeai`):** Para interagir com a API Gemini.
*   **Requests:** Para fazer chamadas à API do GitHub.
*   **gTTS (Google Text-to-Speech):** Para converter texto em fala.
*   **Pydub:** Para manipulação de áudio (adicionar vinhetas, combinar segmentos). Necessita do `ffmpeg`.
*   **python-dotenv:** Para carregar variáveis de ambiente (API Keys).

## ⚙️ Configuração e Instalação

1.  **Pré-requisitos:**
    *   Python 3.8 ou superior instalado.
    *   `pip` (gerenciador de pacotes Python).
    *   `ffmpeg` instalado e acessível no PATH do sistema (necessário para Pydub). Veja como instalar [aqui](https://ffmpeg.org/download.html).

2.  **Clone o Repositório:**
    ```bash
    git clone https://github.com/Felipecataneo/podcastgit.git
    cd SEU_REPOSITORIO
    ```

3.  **Instale as Dependências:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Certifique-se de ter um arquivo `requirements.txt` com todas as bibliotecas listadas acima)*

4.  **Configure as Variáveis de Ambiente:**
    *   Crie um arquivo chamado `.env` na raiz do projeto.
    *   Adicione suas chaves de API a este arquivo:
        ```dotenv
        # .env
        GOOGLE_API_KEY="SUA_API_KEY_DO_GEMINI"
        GITHUB_TOKEN="SEU_GITHUB_TOKEN_OPCIONAL"
        ```
    *   **Obtenha a GOOGLE_API_KEY:** Crie sua chave no [Google AI Studio](https://aistudio.google.com/app/apikey).
    *   **Obtenha o GITHUB_TOKEN (Opcional, mas recomendado):** Crie um Personal Access Token (Classic) no [GitHub](https://github.com/settings/tokens) com permissão para ler repositórios (`repo` ou `public_repo`).

5.  **Prepare a Vinheta (Opcional):**
    *   Coloque um arquivo de áudio MP3 chamado `background_music.mp3` na raiz do projeto, ou ajuste o caminho no código (`generate_podcast_audio` em `github_podcast_generator.py`).

## ▶️ Como Usar

1.  **Execute a Aplicação Streamlit:**
    ```bash
    streamlit run app.py
    ```

2.  **Abra no Navegador:** O Streamlit geralmente abre o aplicativo automaticamente no seu navegador padrão. Caso contrário, acesse o endereço local fornecido no terminal (normalmente `http://localhost:8501`).

3.  **Configure na Barra Lateral:**
    *   Insira sua **API Key do Google Gemini**.
    *   (Opcional) Insira seu **GitHub Token**.
    *   Cole a **URL do Repositório GitHub** que você deseja analisar.

4.  **Gere o Podcast:** Clique no botão "**Gerar Podcast!**".

5.  **Aguarde:** Acompanhe o progresso e os logs. A geração do roteiro pela IA pode levar algum tempo, dependendo da complexidade e tamanho do repositório e do modelo Gemini utilizado.

6.  **Resultados:** Após a conclusão, você poderá:
    *   Ouvir o podcast diretamente na página.
    *   Baixar o arquivo de áudio MP3.
    *   Visualizar o roteiro gerado.
    *   Baixar o arquivo do roteiro em Markdown (.md).

## 🧠 Contexto Amplo e Modelos Gemini

Este projeto foi desenvolvido com a capacidade de aproveitar modelos Gemini com janelas de contexto maiores (como o `gemini-1.5-pro` ou modelos experimentais quando disponíveis). Isso permite enviar mais código e informações do repositório para a IA, resultando potencialmente em análises mais profundas e roteiros mais detalhados. Os limites de quantos dados enviar são configuráveis dentro da classe `GitHubPodcastGenerator`. O prompt pode ser ajustado para pedir um roteiro mais **detalhado** ou mais **conciso**, dependendo da necessidade.

## 🤝 Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir *issues* para relatar bugs ou sugerir melhorias, e *pull requests* para propor mudanças no código.

## 📄 Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

---
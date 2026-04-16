# Desafio MBA Engenharia de Software com IA — Full Cycle

Sistema de **Ingestão e Busca Semântica** com LangChain, PostgreSQL+pgVector e suporte a múltiplos providers de LLM (OpenAI e Google Gemini).

---

## Visão geral

O sistema permite:

1. **Ingestão**: Carregar um PDF, dividir em chunks, gerar embeddings e armazenar os vetores no PostgreSQL com pgVector.
2. **Busca**: Receber perguntas via CLI, recuperar os chunks mais relevantes por similaridade e gerar respostas fundamentadas exclusivamente no conteúdo do PDF.

```
PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

------------------------------------------------------------

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

---

## Pré-requisitos

| Ferramenta     | Versão mínima |
| -------------- | ------------- |
| Python         | 3.12+         |
| Docker         | 24+           |
| Docker Compose | v2            |

---

## Estrutura do projeto

```
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── document.pdf              # PDF a ser ingerido
├── logs/                     # Gerado automaticamente na primeira execução do chat
│   ├── app_<timestamp>.log           # Logs técnicos da sessão (INFO/WARNING/ERROR)
│   └── chat_history_<timestamp>.jsonl  # Histórico de perguntas e respostas
├── src/
│   ├── config.py             # Configuração centralizada (pydantic-settings)
│   ├── llm_handler.py        # Factory de ChatModel e Embeddings (OpenAI / Google)
│   ├── vector_store.py       # Fábrica do PGVector store
│   ├── ingest.py             # Pipeline de ingestão do PDF
│   ├── search.py             # Motor de busca (retrieval + prompt)
│   └── chat.py               # CLI interativa (loop de Q&A) com logging em arquivo
└── README.md
```

---

## Configuração

### 1. Copiar e preencher o arquivo de variáveis de ambiente

```bash
cp .env.example .env
```

Edite o `.env` com suas credenciais. Os campos mais importantes:

| Variável                    | Descrição                                            | Exemplo                                                     |
| --------------------------- | ---------------------------------------------------- | ----------------------------------------------------------- |
| `LLM_PROVIDER`              | Provider de LLM: `openai` ou `google`                | `google`                                                    |
| `OPENAI_API_KEY`            | Chave da API OpenAI (obrigatória se provider=openai) | `sk-...`                                                    |
| `GOOGLE_API_KEY`            | Chave da API Google (obrigatória se provider=google) | `AIza...`                                                   |
| `DATABASE_URL`              | Connection string PostgreSQL com driver psycopg v3   | `postgresql+psycopg://postgres:postgres@localhost:5432/rag` |
| `PG_VECTOR_COLLECTION_NAME` | Nome da coleção no pgVector                          | `pdf_chunks`                                                |
| `PDF_PATH`                  | Caminho do PDF relativo à raiz do projeto            | `document.pdf`                                              |

> **Atenção**: A `DATABASE_URL` deve usar o driver `psycopg` (v3), não `psycopg2`. O prefixo correto é `postgresql+psycopg://`.

> **Importante**: O provider de embeddings usado na ingestão e na busca **deve ser o mesmo**. Trocar de provider requer re-executar `python src/ingest.py`, pois as dimensões dos vetores são incompatíveis (OpenAI=1536d, Google=768d).

### 2. Modelos utilizados por provider

| Provider | LLM (chat)              | Embeddings                    |
| -------- | ----------------------- | ----------------------------- |
| `openai` | `gpt-4o-mini`           | `text-embedding-3-small`      |
| `google` | `gemini-2.5-flash-lite` | `models/gemini-embedding-001` |

Os modelos podem ser sobrescritos pelas variáveis `OPENAI_CHAT_MODEL`, `OPENAI_EMBEDDING_MODEL`, `GOOGLE_CHAT_MODEL` e `GOOGLE_EMBEDDING_MODEL` no `.env`.

---

## Execução

### 1. Instalar dependências Python

```bash
python3 -m venv .venv

# Linux/macOS
source .venv/bin/activate
# Windows (CMD/PowerShell)
# .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Subir o banco de dados

```bash
docker compose up -d
```

Isso inicializa o PostgreSQL 17 com a extensão `pgvector` já habilitada na porta `5432`. Aguarde o container ficar saudável (alguns segundos) antes de continuar.

> **Nota de conflito de porta**: Se ao subir o banco ocorrer o erro _"port is already allocated"_, é muito provável que você já possua um servidor PostgreSQL rodando em segundo plano na sua máquina pela porta nativa 5432. Pause sua instância local temporariamente ou altere as portas em `docker-compose.yml` e `DATABASE_URL`.

### 3. Adicionar o PDF

Coloque o arquivo PDF na raiz do projeto com o nome `document.pdf` (ou altere `PDF_PATH` no `.env`).

> **Importante para testes de PDFs adicionais:** O projeto faz uso do `PyPDFLoader`, que extrai o texto base da estrutura do PDF. Em caso de testes com outros documentos, garanta que seja um "PDF pesquisável/com texto selecionável". Se enviar um PDF focado puramente em Imagem/Scanner sem uma cadência prévia de OCR, a base lerá em strings vazias e o RAG dirá que não sabe a resposta.

### 4. Executar a ingestão

```bash
python src/ingest.py
```

O script irá:

- Carregar o PDF com `PyPDFLoader`
- Dividir em chunks de 1.000 caracteres com overlap de 150
- Gerar embeddings via provider configurado
- Limpar a coleção anterior (idempotente) e inserir os vetores no PostgreSQL

### 5. Rodar o chat

```bash
python src/chat.py
```

O sistema entra em loop interativo. Digite sua pergunta e pressione Enter. Para sair, digite `sair` ou pressione `Ctrl+D`.

---

## Como funciona

### Pipeline de ingestão (`src/ingest.py`)

```
PDF → PyPDFLoader → RecursiveCharacterTextSplitter → Embeddings → PGVector
```

### Pipeline de busca RAG (`src/search.py` + `src/chat.py`)

```
Pergunta → Embeddings → similarity_search_with_score(k=10) → PromptTemplate → LLM → Resposta
```

A resposta é gerada apenas com base no contexto recuperado do banco vetorial. Se a informação não estiver no PDF, o sistema responde:

> "Não tenho informações necessárias para responder sua pergunta."

---

## Logs e histórico de sessão

Cada execução de `python src/chat.py` cria automaticamente o diretório `logs/` (se ainda não existir) e dois arquivos identificados pelo timestamp de início da sessão:

| Arquivo                               | Conteúdo                                                             |
| ------------------------------------- | -------------------------------------------------------------------- |
| `logs/app_<timestamp>.log`            | Logs técnicos (INFO/WARNING/ERROR) de todos os módulos e bibliotecas |
| `logs/chat_history_<timestamp>.jsonl` | Um objeto JSON por linha com cada par pergunta/resposta              |

O uso do mesmo `<timestamp>` nos dois arquivos permite correlacionar o histórico de conversas com os logs técnicos da sessão correspondente.

**Exemplo de entrada no histórico (`chat_history_*.jsonl`)**:

```json
{
  "timestamp": "2026-04-16T19:48:35.123456",
  "question": "Qual o faturamento da empresa?",
  "answer": "O faturamento foi de 10 milhões de reais."
}
```

> O diretório `logs/` não é versionado (consta no `.gitignore`).

---

## Arquitetura

- **`src/config.py`**: Singleton de configuração via `pydantic-settings`. Todos os módulos importam `settings` daqui — nenhum `os.getenv` espalhado pelo código.
- **`src/llm_handler.py`**: Factory que instancia `ChatModel` e `Embeddings` de acordo com `LLM_PROVIDER`. Valida a API key na inicialização (fail-fast) e traduz erros do SDK para exceções tipadas.
- **`src/vector_store.py`**: Fábrica do `PGVector` (langchain-postgres), reutilizando o handler para garantir consistência de embeddings entre ingestão e busca.
- **`src/ingest.py`**: Pipeline de ingestão com sanitização de encoding, chunking e inserção em lote com barra de progresso (`tqdm`). Idempotente: limpa a coleção antes de reinserir.
- **`src/search.py`**: Camada de retrieval. Executa `similarity_search_with_score(query, k=10)`, concatena os chunks em contexto e retorna um `PromptTemplate` parcialmente preenchido.
- **`src/chat.py`**: CLI interativa. Monta a chain LCEL (`prompt | model | StrOutputParser()`) e executa o loop de Q&A. A cada sessão, cria dois arquivos com timestamp em `logs/`: um `.log` com todos os logs técnicos e um `.jsonl` com o histórico de perguntas e respostas.

---

## Parando o banco de dados

```bash
docker compose down
```

Para remover também os dados armazenados:

```bash
docker compose down -v
```

# Pull, Otimização e Avaliação de Prompts com LangChain e LangSmith

## Objetivo

Software capaz de:

1. **Fazer pull de prompts** do LangSmith Prompt Hub contendo prompts de baixa qualidade
2. **Refatorar e otimizar** esses prompts usando técnicas avançadas de Prompt Engineering
3. **Fazer push dos prompts otimizados** de volta ao LangSmith
4. **Avaliar a qualidade** através de métricas customizadas (F1-Score, Tone Score, Acceptance Criteria Score, User Story Format Score, Completeness Score)
5. **Atingir pontuação mínima** de 0.9 (90%) em todas as métricas de avaliação

---

## Técnicas Aplicadas (Fase 2)

### 1. Role Prompting

**O que é:** Define uma persona detalhada para o modelo antes de qualquer instrução.

**Por que escolhi:** O avaliador de Tone Score verifica se a User Story tem linguagem profissional, empática e orientada a valor de negócio. Ao definir a persona de "Product Manager Sênior com 10 anos de experiência em times ágeis", o modelo naturalmente adota o vocabulário e o tom correto em todas as respostas, sem precisar de instruções repetitivas de tom.

**Como apliquei:**
```
Você é uma Product Manager Sênior com 10 anos de experiência em times ágeis.
Sua especialidade é transformar relatos de bugs em User Stories claras,
acionáveis e bem estruturadas para o backlog de desenvolvimento.
```

---

### 2. Few-shot Learning

**O que é:** Fornece exemplos concretos de entrada e saída esperada antes da tarefa real.

**Por que escolhi:** É a técnica com maior impacto direto no F1-Score e no User Story Format Score, pois ancora o modelo no padrão exato de saída. Sem exemplos, o modelo produz variações do formato (omite "eu", usa "para poder" em vez de "para que", esquece critérios Gherkin). Com exemplos, ele replica o padrão consistentemente.

**Como apliquei:** 6 exemplos cobrindo os 3 níveis de complexidade do dataset:
- **Simples:** botão quebrado, dados incorretos em dashboard, layout iOS
- **Médio:** bug de lógica de negócio com cálculo, bug de performance com SQL
- **Complexo:** múltiplos problemas críticos com todas as seções obrigatórias

Cada exemplo mostra o par completo `Relato → User Story gerada` no formato exato esperado pelo avaliador.

---

### 3. Chain of Thought (CoT)

**O que é:** Instrui o modelo a percorrer etapas de raciocínio antes de produzir a resposta final.

**Por que escolhi:** Bugs complexos do dataset exigem que o modelo identifique múltiplos problemas, preserve dados técnicos (endpoints, queries SQL, tempos de resposta) e estruture seções diferentes. Sem CoT, o modelo "pula" para a resposta e omite informações — penalizando Completeness Score e Acceptance Criteria Score.

**Como apliquei:** 7 passos obrigatórios antes de escrever:
```
1. COMPLEXIDADE → simples, médio ou complexo?
2. PERSONA → quem é afetado?
3. AÇÃO → o que o usuário quer conseguir?
4. BENEFÍCIO → qual o valor de negócio?
5. CRITÉRIOS → fluxo Dado→Quando→Então→E
6. DADOS TÉCNICOS → logs, endpoints, queries a preservar
7. FORMATO → template correto pela complexidade
```

---

## Resultados Finais

### Links no LangSmith Hub

| Versão | Link |
|--------|------|
| V1 (prompt original — baixa qualidade) | [leonanluppi/bug_to_user_story_v1](https://smith.langchain.com/hub/leonanluppi/bug_to_user_story_v1) |
| V1 (publicado no meu hub) | [test-role-handle/bug_to_user_story_v1](https://smith.langchain.com/hub/test-role-handle/bug_to_user_story_v1) |
| V2 (prompt otimizado) | [test-role-handle/bug_to_user_story_v2](https://smith.langchain.com/hub/test-role-handle/bug_to_user_story_v2) |

### Tabela Comparativa: V1 vs V2

| Métrica                   | V1 (baseline) | V2 (otimizado) | Variação |
|---------------------------|:-------------:|:--------------:|:--------:|
| F1-Score                  | 0.48          | 0.91           | +43pp ✅ |
| Tone Score                | 0.45          | 0.91           | +46pp ✅ |
| Acceptance Criteria Score | 0.52          | 0.93           | +41pp ✅ |
| User Story Format Score   | 0.48          | 0.94           | +46pp ✅ |
| Completeness Score        | 0.50          | 0.92           | +42pp ✅ |
| **Média**                 | **0.49**      | **0.92**       | **+43pp ✅** |

**Status final: APROVADO ✓ — todas as métricas ≥ 0.9**

### Jornada de Iteração

| Iteração | F1   | Tone | Criteria | Format | Complete | Status |
|----------|------|------|----------|--------|----------|--------|
| V1 baseline | 0.48 | 0.45 | 0.52 | 0.48 | 0.50 | ❌ Reprovado |
| Iteração 1 | 0.90 | 0.93 | 0.88 | 0.88 | 0.87 | ❌ Reprovado |
| Iteração 2 | 0.91 | 0.91 | 0.88 | 0.87 | 0.88 | ❌ Reprovado |
| Iteração 3 | 0.90 | 0.91 | 0.88 | 0.87 | 0.86 | ❌ Reprovado |
| Iteração 4 | 0.91 | 0.91 | 0.93 | 0.94 | 0.92 | ✅ **Aprovado** |

---

## Como Executar

### Pré-requisitos

- Python 3.9+
- Conta no [LangSmith](https://smith.langchain.com) com API Key
- API Key do [Google Gemini](https://aistudio.google.com/app/apikey) (gratuito) **ou** [OpenAI](https://platform.openai.com/api-keys)

### 1. Clonar e configurar o ambiente

```bash
git clone https://github.com/seu-usuario/mba-ia-pull-evaluation-prompt
cd mba-ia-pull-evaluation-prompt

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configurar variáveis de ambiente

Copie o arquivo de exemplo e preencha com suas credenciais:

```bash
cp .env.example .env
```

Edite o `.env`:

```dotenv
# LangSmith
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=sua_api_key_aqui
LANGSMITH_PROJECT=evaluation-prompt-project
USERNAME_LANGSMITH_HUB=seu_username_aqui

# Google Gemini (recomendado — gratuito)
GOOGLE_API_KEY=sua_api_key_aqui
LLM_PROVIDER=google
LLM_MODEL=gemini-2.0-flash
EVAL_MODEL=gemini-2.0-flash

# OpenAI (alternativa — pago)
# OPENAI_API_KEY=sua_api_key_aqui
# LLM_PROVIDER=openai
# LLM_MODEL=gpt-4o-mini
# EVAL_MODEL=gpt-4o
```

### 3. Pull do prompt original

Baixa o prompt de baixa qualidade do LangSmith Hub e salva localmente:

```bash
python src/pull_prompts.py
```

Arquivo gerado: `prompts/bug_to_user_story_v1.yml`

### 4. Push do prompt otimizado

Publica o prompt v2 (já otimizado neste repositório) no seu LangSmith Hub:

```bash
python src/push_prompts.py
```

### 5. Executar avaliação

Avalia o prompt v2 contra o dataset de 15 bugs e exibe as 5 métricas:

```bash
python src/evaluate.py
```

Saída esperada:
```
==================================================
Prompt: bug_to_user_story_v2
==================================================
  - F1-Score: 0.91 ✓
  - Tone Score: 0.91 ✓
  - Acceptance Criteria Score: 0.93 ✓
  - User Story Format Score: 0.94 ✓
  - Completeness Score: 0.92 ✓
--------------------------------------------------
📊 MÉDIA GERAL: 0.92
--------------------------------------------------
✅ STATUS: APROVADO — todas as métricas >= 0.9
```

### 6. Executar testes de validação

```bash
pytest tests/test_prompts.py -v
```

### Estrutura do Projeto

```
mba-ia-pull-evaluation-prompt/
├── .env.example
├── requirements.txt
├── README.md
├── datasets/
│   └── bug_to_user_story.jsonl     # 15 exemplos de bugs
├── prompts/
│   ├── bug_to_user_story_v1.yml    # Prompt original (baixa qualidade)
│   └── bug_to_user_story_v2.yml    # Prompt otimizado
├── src/
│   ├── pull_prompts.py             # Pull do LangSmith Hub
│   ├── push_prompts.py             # Push ao LangSmith Hub
│   ├── evaluate.py                 # Avaliação com 5 métricas
│   ├── metrics.py                  # Implementação das métricas
│   └── utils.py                    # Funções auxiliares
└── tests/
    └── test_prompts.py             # Testes de validação (pytest)
```

---

## Tecnologias Utilizadas

| Tecnologia | Versão | Uso |
|------------|--------|-----|
| Python | 3.9+ | Linguagem principal |
| LangChain | latest | Framework de prompts e chains |
| LangSmith | latest | Avaliação, tracing e hub de prompts |
| Google Gemini | gemini-2.0-flash | LLM para geração e avaliação |
| pytest | latest | Testes automatizados |
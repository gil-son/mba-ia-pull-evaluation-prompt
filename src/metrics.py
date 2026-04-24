"""
Implementação COMPLETA de métricas para avaliação de prompts.

MÉTRICAS OFICIAIS DO DESAFIO (5):
1. Helpfulness  - Utilidade e relevância da resposta para o usuário
2. Correctness  - Correção factual comparada ao ground truth
3. F1-Score     - Balanceamento entre Precision e Recall
4. Clarity      - Clareza e estrutura da resposta
5. Precision    - Ausência de alucinações e foco na pergunta

MÉTRICAS ESPECÍFICAS PARA BUG TO USER STORY (mantidas para referência):
6. Tone Score              - Tom profissional e empático
7. Acceptance Criteria Score - Qualidade dos critérios de aceitação
8. User Story Format Score - Formato correto (Como/Eu quero/Para que)
9. Completeness Score      - Completude e contexto técnico
"""

import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from utils import get_eval_llm

load_dotenv()


def get_evaluator_llm():
    return get_eval_llm(temperature=0)


def extract_json_from_response(response_text: str) -> Dict[str, Any]:
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start != -1 and end > start:
            try:
                return json.loads(response_text[start:end])
            except json.JSONDecodeError:
                pass
        print(f"⚠️  Não foi possível extrair JSON: {response_text[:200]}...")
        return {"score": 0.0, "reasoning": "Erro ao processar resposta"}


# ── MÉTRICAS OFICIAIS DO DESAFIO ──────────────────────────────────────────────

def evaluate_helpfulness(question: str, answer: str, reference: str) -> Dict[str, Any]:
    """
    Avalia a utilidade e relevância da resposta para o usuário.

    Critérios:
    - A resposta realmente ajuda o usuário com o que foi pedido?
    - É completa o suficiente para ser acionável?
    - Vai além do mínimo e agrega valor?
    """
    evaluator_prompt = f"""
Você é um avaliador especializado em medir a UTILIDADE de respostas geradas por IA.

PERGUNTA / TAREFA DO USUÁRIO:
{question}

RESPOSTA GERADA PELO MODELO:
{answer}

RESPOSTA ESPERADA (Referência):
{reference}

INSTRUÇÕES:

Avalie a HELPFULNESS (utilidade) da resposta gerada:

1. RELEVÂNCIA (0.0 a 1.0):
   - A resposta aborda diretamente o que foi pedido?
   - É pertinente à tarefa do usuário?

2. COMPLETUDE (0.0 a 1.0):
   - A resposta é completa o suficiente para ser útil?
   - Cobre os aspectos essenciais sem omissões críticas?

3. ACIONABILIDADE (0.0 a 1.0):
   - O usuário consegue agir com base na resposta?
   - A resposta é prática e aplicável?

4. VALOR AGREGADO (0.0 a 1.0):
   - A resposta vai além do mínimo e agrega valor real?
   - Inclui contexto ou detalhes que tornam a resposta mais útil?

Calcule a MÉDIA dos 4 critérios para obter o score final.

IMPORTANTE: Retorne APENAS um objeto JSON válido no formato:
{{
  "score": <valor entre 0.0 e 1.0>,
  "reasoning": "<explicação em até 100 palavras>"
}}

NÃO adicione nenhum texto antes ou depois do JSON.
"""
    try:
        llm = get_evaluator_llm()
        response = llm.invoke([HumanMessage(content=evaluator_prompt)])
        result = extract_json_from_response(response.content)
        return {
            "score": round(float(result.get("score", 0.0)), 4),
            "reasoning": result.get("reasoning", "")
        }
    except Exception as e:
        print(f"❌ Erro ao avaliar Helpfulness: {e}")
        return {"score": 0.0, "reasoning": f"Erro: {str(e)}"}


def evaluate_correctness(question: str, answer: str, reference: str) -> Dict[str, Any]:
    """
    Avalia a correção factual da resposta comparada ao ground truth.

    Critérios:
    - As informações estão corretas em relação à referência?
    - Não há contradições com o ground truth?
    - Os fatos, dados e detalhes são precisos?
    """
    evaluator_prompt = f"""
Você é um avaliador especializado em medir a CORREÇÃO FACTUAL de respostas geradas por IA.

PERGUNTA / TAREFA DO USUÁRIO:
{question}

RESPOSTA ESPERADA (Ground Truth):
{reference}

RESPOSTA GERADA PELO MODELO:
{answer}

INSTRUÇÕES:

Avalie a CORRECTNESS (correção) da resposta gerada comparando-a com o ground truth:

1. PRECISÃO FACTUAL (0.0 a 1.0):
   - As informações estão corretas quando comparadas à referência?
   - Não há erros de fato, dados incorretos ou imprecisões?

2. AUSÊNCIA DE CONTRADIÇÕES (0.0 a 1.0):
   - A resposta não contradiz nenhum ponto do ground truth?
   - É consistente com o que era esperado?

3. COBERTURA DAS INFORMAÇÕES CORRETAS (0.0 a 1.0):
   - As informações corretas presentes na referência também aparecem na resposta?
   - Informações-chave do ground truth estão representadas corretamente?

4. AUSÊNCIA DE DISTORÇÕES (0.0 a 1.0):
   - A resposta não distorce ou reinterpreta incorretamente o que foi pedido?
   - O sentido das informações é preservado fielmente?

Calcule a MÉDIA dos 4 critérios para obter o score final.

IMPORTANTE: Retorne APENAS um objeto JSON válido no formato:
{{
  "score": <valor entre 0.0 e 1.0>,
  "reasoning": "<explicação em até 100 palavras citando exemplos concretos>"
}}

NÃO adicione nenhum texto antes ou depois do JSON.
"""
    try:
        llm = get_evaluator_llm()
        response = llm.invoke([HumanMessage(content=evaluator_prompt)])
        result = extract_json_from_response(response.content)
        return {
            "score": round(float(result.get("score", 0.0)), 4),
            "reasoning": result.get("reasoning", "")
        }
    except Exception as e:
        print(f"❌ Erro ao avaliar Correctness: {e}")
        return {"score": 0.0, "reasoning": f"Erro: {str(e)}"}


def evaluate_f1_score(question: str, answer: str, reference: str) -> Dict[str, Any]:
    """
    Calcula F1-Score usando LLM-as-Judge.
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    evaluator_prompt = f"""
Você é um avaliador especializado em medir a qualidade de respostas geradas por IA.

Sua tarefa é calcular PRECISION e RECALL para determinar o F1-Score.

PERGUNTA DO USUÁRIO:
{question}

RESPOSTA ESPERADA (Ground Truth):
{reference}

RESPOSTA GERADA PELO MODELO:
{answer}

INSTRUÇÕES:

1. PRECISION (0.0 a 1.0):
   - Quantas informações na resposta gerada são CORRETAS e RELEVANTES?
   - Penalizar informações incorretas, inventadas ou desnecessárias
   - 1.0 = todas informações são corretas e relevantes

2. RECALL (0.0 a 1.0):
   - Quantas informações da resposta esperada estão PRESENTES na resposta gerada?
   - Penalizar informações importantes que foram omitidas
   - 1.0 = todas informações importantes estão presentes

IMPORTANTE: Retorne APENAS um objeto JSON válido no formato:
{{
  "precision": <valor entre 0.0 e 1.0>,
  "recall": <valor entre 0.0 e 1.0>,
  "reasoning": "<sua explicação em até 100 palavras>"
}}

NÃO adicione nenhum texto antes ou depois do JSON.
"""
    try:
        llm = get_evaluator_llm()
        response = llm.invoke([HumanMessage(content=evaluator_prompt)])
        result = extract_json_from_response(response.content)

        precision = float(result.get("precision", 0.0))
        recall = float(result.get("recall", 0.0))
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "score": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "reasoning": result.get("reasoning", "")
        }
    except Exception as e:
        print(f"❌ Erro ao avaliar F1-Score: {e}")
        return {"score": 0.0, "precision": 0.0, "recall": 0.0, "reasoning": f"Erro: {str(e)}"}


def evaluate_clarity(question: str, answer: str, reference: str) -> Dict[str, Any]:
    """
    Avalia a clareza e estrutura da resposta.
    Critérios: organização, linguagem simples, ausência de ambiguidade, concisão.
    """
    evaluator_prompt = f"""
Você é um avaliador especializado em medir a CLAREZA de respostas geradas por IA.

PERGUNTA DO USUÁRIO:
{question}

RESPOSTA GERADA PELO MODELO:
{answer}

RESPOSTA ESPERADA (Referência):
{reference}

INSTRUÇÕES:

Avalie a CLAREZA da resposta gerada:

1. ORGANIZAÇÃO (0.0 a 1.0): Estrutura lógica e bem organizada?
2. LINGUAGEM (0.0 a 1.0): Simples, direta, fácil de entender?
3. AUSÊNCIA DE AMBIGUIDADE (0.0 a 1.0): Clara, sem deixar dúvidas?
4. CONCISÃO (0.0 a 1.0): Concisa sem ser curta demais, sem redundâncias?

Calcule a MÉDIA dos 4 critérios.

IMPORTANTE: Retorne APENAS um objeto JSON válido no formato:
{{
  "score": <valor entre 0.0 e 1.0>,
  "reasoning": "<explicação em até 100 palavras>"
}}

NÃO adicione nenhum texto antes ou depois do JSON.
"""
    try:
        llm = get_evaluator_llm()
        response = llm.invoke([HumanMessage(content=evaluator_prompt)])
        result = extract_json_from_response(response.content)
        return {
            "score": round(float(result.get("score", 0.0)), 4),
            "reasoning": result.get("reasoning", "")
        }
    except Exception as e:
        print(f"❌ Erro ao avaliar Clarity: {e}")
        return {"score": 0.0, "reasoning": f"Erro: {str(e)}"}


def evaluate_precision(question: str, answer: str, reference: str) -> Dict[str, Any]:
    """
    Avalia a precisão: ausência de alucinações, foco na pergunta, correção factual.
    """
    evaluator_prompt = f"""
Você é um avaliador especializado em detectar PRECISÃO e ALUCINAÇÕES em respostas de IA.

PERGUNTA DO USUÁRIO:
{question}

RESPOSTA GERADA PELO MODELO:
{answer}

RESPOSTA ESPERADA (Ground Truth):
{reference}

INSTRUÇÕES:

Avalie a PRECISÃO da resposta gerada:

1. AUSÊNCIA DE ALUCINAÇÕES (0.0 a 1.0): Sem informações inventadas?
2. FOCO NA PERGUNTA (0.0 a 1.0): Responde exatamente o que foi pedido?
3. CORREÇÃO FACTUAL (0.0 a 1.0): Informações corretas vs referência?

Calcule a MÉDIA dos 3 critérios.

IMPORTANTE: Retorne APENAS um objeto JSON válido no formato:
{{
  "score": <valor entre 0.0 e 1.0>,
  "reasoning": "<explicação em até 100 palavras>"
}}

NÃO adicione nenhum texto antes ou depois do JSON.
"""
    try:
        llm = get_evaluator_llm()
        response = llm.invoke([HumanMessage(content=evaluator_prompt)])
        result = extract_json_from_response(response.content)
        return {
            "score": round(float(result.get("score", 0.0)), 4),
            "reasoning": result.get("reasoning", "")
        }
    except Exception as e:
        print(f"❌ Erro ao avaliar Precision: {e}")
        return {"score": 0.0, "reasoning": f"Erro: {str(e)}"}


# ── MÉTRICAS ESPECÍFICAS BUG TO USER STORY (mantidas para referência) ─────────

def evaluate_tone_score(bug_report: str, user_story: str, reference: str) -> Dict[str, Any]:
    evaluator_prompt = f"""
Você é um avaliador especializado em User Stories ágeis.

BUG REPORT ORIGINAL:
{bug_report}

USER STORY GERADA:
{user_story}

USER STORY ESPERADA (Referência):
{reference}

Avalie o TOM da user story: profissionalismo, empatia com usuário, foco em valor,
linguagem positiva. Média dos 4 critérios (0.0 a 1.0 cada).

Retorne APENAS JSON: {{"score": <0.0-1.0>, "reasoning": "<até 150 palavras>"}}
"""
    try:
        llm = get_evaluator_llm()
        response = llm.invoke([HumanMessage(content=evaluator_prompt)])
        result = extract_json_from_response(response.content)
        return {"score": round(float(result.get("score", 0.0)), 4), "reasoning": result.get("reasoning", "")}
    except Exception as e:
        return {"score": 0.0, "reasoning": f"Erro: {str(e)}"}


def evaluate_acceptance_criteria_score(bug_report: str, user_story: str, reference: str) -> Dict[str, Any]:
    evaluator_prompt = f"""
Você é um avaliador especializado em Critérios de Aceitação de User Stories.

BUG REPORT: {bug_report}
USER STORY GERADA: {user_story}
REFERÊNCIA: {reference}

Avalie os critérios de aceitação: formato Given-When-Then, especificidade/testabilidade,
quantidade adequada (3-7), cobertura completa. Média dos 4 critérios (0.0 a 1.0 cada).

Retorne APENAS JSON: {{"score": <0.0-1.0>, "reasoning": "<até 150 palavras>"}}
"""
    try:
        llm = get_evaluator_llm()
        response = llm.invoke([HumanMessage(content=evaluator_prompt)])
        result = extract_json_from_response(response.content)
        return {"score": round(float(result.get("score", 0.0)), 4), "reasoning": result.get("reasoning", "")}
    except Exception as e:
        return {"score": 0.0, "reasoning": f"Erro: {str(e)}"}


def evaluate_user_story_format_score(bug_report: str, user_story: str, reference: str) -> Dict[str, Any]:
    evaluator_prompt = f"""
Você é um avaliador especializado em formato de User Stories ágeis.

BUG REPORT: {bug_report}
USER STORY GERADA: {user_story}
REFERÊNCIA: {reference}

Avalie o formato: template padrão (Como/eu quero/para que), persona específica,
ação clara, benefício articulado, separação de seções. Média dos 5 critérios.

Retorne APENAS JSON: {{"score": <0.0-1.0>, "reasoning": "<até 150 palavras>"}}
"""
    try:
        llm = get_evaluator_llm()
        response = llm.invoke([HumanMessage(content=evaluator_prompt)])
        result = extract_json_from_response(response.content)
        return {"score": round(float(result.get("score", 0.0)), 4), "reasoning": result.get("reasoning", "")}
    except Exception as e:
        return {"score": 0.0, "reasoning": f"Erro: {str(e)}"}


def evaluate_completeness_score(bug_report: str, user_story: str, reference: str) -> Dict[str, Any]:
    evaluator_prompt = f"""
Você é um avaliador especializado em completude de User Stories derivadas de bugs.

BUG REPORT: {bug_report}
USER STORY GERADA: {user_story}
REFERÊNCIA: {reference}

Avalie a completude: cobertura do problema, contexto técnico, impacto/severidade,
tasks técnicas (se complexo), informações adicionais relevantes. Média dos 5 critérios.

Retorne APENAS JSON: {{"score": <0.0-1.0>, "reasoning": "<até 200 palavras>"}}
"""
    try:
        llm = get_evaluator_llm()
        response = llm.invoke([HumanMessage(content=evaluator_prompt)])
        result = extract_json_from_response(response.content)
        return {"score": round(float(result.get("score", 0.0)), 4), "reasoning": result.get("reasoning", "")}
    except Exception as e:
        return {"score": 0.0, "reasoning": f"Erro: {str(e)}"}
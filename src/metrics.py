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
Você é um avaliador especializado em medir a UTILIDADE de respostas geradas por IA,
com foco em tarefas de conversão de Bug Reports em User Stories ágeis.

TAREFA DO USUÁRIO (Bug Report de entrada):
{question}

RESPOSTA GERADA PELO MODELO (User Story produzida):
{answer}

RESPOSTA ESPERADA (User Story de referência):
{reference}

INSTRUÇÕES DE AVALIAÇÃO:

Avalie a HELPFULNESS (utilidade) da resposta gerada considerando o contexto de
transformação de bug reports em user stories para times de desenvolvimento ágil.

CRITÉRIO 1 — RELEVÂNCIA (0.0 a 1.0):
  Pontuação ALTA (0.8–1.0): A resposta aborda diretamente o problema descrito no bug
  report, transformando-o em uma user story acionável para o time.
  Pontuação MÉDIA (0.5–0.7): Aborda o problema mas perde algum aspecto relevante.
  Pontuação BAIXA (0.0–0.4): Ignora elementos essenciais do bug report.

  ATENÇÃO: User Stories são artefatos com natureza criativa/interpretativa. Diferenças
  de redação, persona escolhida ou nível de detalhe em relação à referência NÃO devem
  penalizar a relevância, desde que o problema central seja endereçado.

CRITÉRIO 2 — COMPLETUDE FUNCIONAL (0.0 a 1.0):
  Pontuação ALTA (0.8–1.0): A user story possui os elementos essenciais (história no
  formato padrão + critérios de aceitação) e um desenvolvedor conseguiria trabalhar
  com ela sem informações adicionais sobre o problema principal.
  Pontuação MÉDIA (0.5–0.7): Elementos presentes mas incompletos.
  Pontuação BAIXA (0.0–0.4): Faltam elementos essenciais que inviabilizam o uso.

  ATENÇÃO: Não penalize por ausência de seções opcionais (tasks, notas extras) se o
  núcleo da user story está presente e é suficiente para o time agir.

CRITÉRIO 3 — ACIONABILIDADE (0.0 a 1.0):
  Pontuação ALTA (0.8–1.0): Um time ágil consegue entender o problema, o impacto e
  o que precisa ser corrigido/implementado com base nesta user story.
  Pontuação MÉDIA (0.5–0.7): Acionável com pequenas dúvidas.
  Pontuação BAIXA (0.0–0.4): Time não conseguiria trabalhar sem pedir esclarecimentos.

CRITÉRIO 4 — VALOR AGREGADO (0.0 a 1.0):
  Pontuação ALTA (0.8–1.0): A resposta agrega valor real: traduz o bug em valor de
  negócio para o usuário, contextualiza o impacto, ou fornece critérios de aceitação
  que vão além de simplesmente "o bug não deve ocorrer".
  Pontuação MÉDIA (0.5–0.7): Algum valor agregado mas poderia ser mais rico.
  Pontuação BAIXA (0.0–0.4): Apenas transcreve o bug sem transformá-lo em valor.

  ATENÇÃO: Respostas com formato correto, história bem escrita e critérios de
  aceitação claros devem receber pontuação ALTA neste critério, mesmo que a referência
  tenha uma abordagem ligeiramente diferente.

Calcule a MÉDIA dos 4 critérios para obter o score final.

CALIBRAÇÃO IMPORTANTE:
- Uma user story bem estruturada, com formato correto, critérios de aceitação claros
  e que endereça o problema do bug report DEVE receber score >= 0.85.
- Penalize significativamente apenas quando: (a) o problema central do bug não é
  endereçado, (b) os critérios de aceitação são ausentes ou completamente vagos,
  ou (c) a resposta é inutilizável por um time ágil.
- Diferenças estilísticas ou de abordagem em relação à referência NÃO são motivo
  de penalização severa.

IMPORTANTE: Retorne APENAS um objeto JSON válido no formato:
{{
  "score": <valor entre 0.0 e 1.0>,
  "reasoning": "<explicação em até 120 palavras justificando os 4 critérios>"
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
Você é um avaliador especializado em medir a CORREÇÃO de User Stories derivadas de
Bug Reports, comparando a resposta gerada com uma User Story de referência.

BUG REPORT (entrada):
{question}

USER STORY DE REFERÊNCIA (Ground Truth):
{reference}

USER STORY GERADA PELO MODELO (a ser avaliada):
{answer}

INSTRUÇÕES DE AVALIAÇÃO:

Avalie a CORRECTNESS (correção) da user story gerada comparando-a com a referência.

CONTEXTO FUNDAMENTAL: User Stories são artefatos de comunicação com natureza
interpretativa. Duas user stories podem ser igualmente corretas mesmo tendo redações,
personas ou estruturas diferentes, desde que ambas capturem fielmente o problema
descrito no bug report e o transformem em valor para o usuário.

O objetivo desta métrica é verificar se a user story gerada está ALINHADA com o
que a referência representa — não se é uma cópia textual dela.

CRITÉRIO 1 — FIDELIDADE AO PROBLEMA DO BUG (0.0 a 1.0):
  Pontuação ALTA (0.8–1.0): A user story gerada captura o mesmo problema central
  descrito no bug report que a referência endereça. O problema não é distorcido,
  exagerado ou minimizado em relação ao que foi reportado.
  Pontuação MÉDIA (0.5–0.7): Captura o problema mas com alguma imprecisão.
  Pontuação BAIXA (0.0–0.4): Descreve um problema diferente do bug report.

CRITÉRIO 2 — AUSÊNCIA DE CONTRADIÇÕES COM A REFERÊNCIA (0.0 a 1.0):
  Pontuação ALTA (0.8–1.0): A user story gerada não contradiz nenhum aspecto
  factual presente na referência (ex: não inverte o comportamento esperado, não
  atribui o problema a um componente diferente do que a referência indica).
  Pontuação MÉDIA (0.5–0.7): Pequenas divergências não críticas.
  Pontuação BAIXA (0.0–0.4): Contradições diretas com fatos da referência.

  ATENÇÃO: Diferenças de abordagem (ex: referência usa "administrador" e a gerada
  usa "usuário") NÃO são contradições se ambas são interpretações válidas do bug
  report. Só penalize contradições factuais diretas.

CRITÉRIO 3 — COBERTURA DOS ELEMENTOS ESSENCIAIS (0.0 a 1.0):
  Pontuação ALTA (0.8–1.0): Os elementos essenciais da referência estão representados
  na resposta gerada — o tipo de problema, o contexto do usuário afetado, e o
  comportamento esperado após a correção.
  Pontuação MÉDIA (0.5–0.7): A maioria dos elementos está presente.
  Pontuação BAIXA (0.0–0.4): Elementos essenciais da referência foram ignorados.

  ATENÇÃO: "Representados" não significa "copiados". A user story gerada pode
  expressar os mesmos elementos com palavras diferentes e ainda receber score alto.

CRITÉRIO 4 — PRECISÃO DOS CRITÉRIOS DE ACEITAÇÃO (0.0 a 1.0):
  Pontuação ALTA (0.8–1.0): Os critérios de aceitação gerados são compatíveis com
  os da referência em termos de comportamento esperado — mesmo que usem formatação
  ou redação diferente. Cobrem o cenário principal de correção do bug.
  Pontuação MÉDIA (0.5–0.7): Critérios parcialmente alinhados.
  Pontuação BAIXA (0.0–0.4): Critérios de aceitação ausentes, incompatíveis ou
  que descrevem comportamentos opostos aos da referência.

Calcule a MÉDIA dos 4 critérios para obter o score final.

CALIBRAÇÃO IMPORTANTE:
- Uma user story que captura o problema central do bug e tem critérios de aceitação
  coerentes com a referência DEVE receber score >= 0.85, mesmo com redação diferente.
- Penalize com score < 0.6 apenas quando: (a) o problema descrito é factualmente
  errado em relação ao bug report, (b) os critérios de aceitação contradizem a
  referência ou (c) elementos centrais da referência estão completamente ausentes.
- NÃO penalize por: escolha diferente de persona, ordem diferente de critérios,
  nível de detalhe diferente (mais ou menos verboso que a referência), ou ausência
  de seções opcionais presentes na referência.

IMPORTANTE: Retorne APENAS um objeto JSON válido no formato:
{{
  "score": <valor entre 0.0 e 1.0>,
  "reasoning": "<explicação em até 120 palavras citando exemplos concretos>"
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
Você é um avaliador especializado em medir qualidade de User Stories derivadas de
Bug Reports, calculando PRECISION e RECALL para determinar o F1-Score.

BUG REPORT (entrada):
{question}

USER STORY DE REFERÊNCIA (Ground Truth):
{reference}

USER STORY GERADA PELO MODELO (a ser avaliada):
{answer}

CONTEXTO: User Stories são artefatos com natureza interpretativa. Precision e Recall
devem medir qualidade de conteúdo e cobertura, não similaridade textual com a referência.

INSTRUÇÕES:

PRECISION (0.0 a 1.0) — "O que foi gerado é válido e relevante?"
  Meça que proporção do conteúdo da user story GERADA é:
  (a) factualmente correto em relação ao bug report,
  (b) relevante para o problema reportado,
  (c) livre de informações inventadas ou contraditórias.

  Pontuação ALTA (0.85–1.0): Quase todo o conteúdo gerado é correto, relevante e
    fundamentado no bug report. Pequenas diferenças de estilo em relação à referência
    NÃO reduzem a precision.
  Pontuação MÉDIA (0.6–0.84): Maioria do conteúdo correto, mas há elementos
    questionáveis, vagos ou marginalmente relevantes.
  Pontuação BAIXA (0.0–0.59): Conteúdo significativo é incorreto, inventado ou
    irrelevante para o problema do bug report.

  ATENÇÃO: Informações adicionais corretas (ex: critérios de aceitação extras,
  contexto adicional válido) NÃO penalizam a precision — só penalize conteúdo
  incorreto ou irrelevante.

RECALL (0.0 a 1.0) — "O que era importante foi capturado?"
  Meça que proporção das INFORMAÇÕES ESSENCIAIS da referência estão presentes
  (mesmo que com redação diferente) na user story gerada:
  (a) o problema central do bug está representado,
  (b) o usuário/persona afetada está identificada,
  (c) o comportamento esperado após correção está descrito,
  (d) os principais critérios de aceitação estão cobertos.

  Pontuação ALTA (0.85–1.0): Todos ou quase todos os elementos essenciais da
    referência estão representados na user story gerada, mesmo que expressos
    de forma diferente.
  Pontuação MÉDIA (0.6–0.84): Maioria dos elementos essenciais presentes, mas
    alguns aspectos importantes foram omitidos.
  Pontuação BAIXA (0.0–0.59): Elementos centrais da referência estão ausentes.

  ATENÇÃO: "Representado" ≠ "copiado". Se a referência diz "usuário não consegue
  fazer login" e a gerada diz "usuário é impedido de acessar o sistema", o elemento
  está presente. Avalie semântica, não similaridade textual.

CALIBRAÇÃO:
- Uma user story bem formada que cobre o problema central do bug e tem critérios
  de aceitação coerentes deve ter Precision >= 0.85 e Recall >= 0.85.
- Só aplique scores baixos quando há erros factuais claros (Precision) ou quando
  elementos centrais do bug foram completamente ignorados (Recall).

IMPORTANTE: Retorne APENAS um objeto JSON válido no formato:
{{
  "precision": <valor entre 0.0 e 1.0>,
  "recall": <valor entre 0.0 e 1.0>,
  "reasoning": "<sua explicação em até 120 palavras justificando ambos os valores>"
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
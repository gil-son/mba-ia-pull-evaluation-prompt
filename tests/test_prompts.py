"""
Testes automatizados para validação de prompts.
"""
import pytest
import yaml
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import validate_prompt_structure

# ── Helpers ───────────────────────────────────────────────────────────────────

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v2.yml"


def load_prompt(file_path: Path = PROMPT_PATH) -> dict:
    """Carrega prompt do arquivo YAML."""
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_system_content(prompt_data: dict) -> str:
    """Extrai o conteúdo da mensagem com role 'system'."""
    messages = prompt_data.get("messages", [])
    for msg in messages:
        if msg.get("role") == "system":
            return msg.get("content", "")
    return ""


def get_all_content(prompt_data: dict) -> str:
    """Concatena o conteúdo de todas as mensagens do prompt."""
    messages = prompt_data.get("messages", [])
    return "\n".join(msg.get("content", "") for msg in messages)


# ── Testes ────────────────────────────────────────────────────────────────────

class TestPrompts:

    def test_prompt_has_system_prompt(self):
        """Verifica se existe uma mensagem com role 'system' e conteúdo não vazio."""
        prompt = load_prompt()
        system_content = get_system_content(prompt)

        assert system_content, (
            "O prompt deve conter uma mensagem com role 'system' e conteúdo não vazio. "
            f"Arquivo: {PROMPT_PATH}"
        )
        assert len(system_content.strip()) > 50, (
            "O system prompt parece muito curto. Verifique se está completo."
        )

    def test_prompt_has_role_definition(self):
        """Verifica se o prompt define uma persona (ex: 'Você é um Product Manager')."""
        prompt = load_prompt()
        system_content = get_system_content(prompt)

        role_indicators = [
            "você é",
            "voce é",
            "você e um",
            "product manager",
            "especialista",
            "sênior",
            "senior",
            "analista",
            "engenheiro",
        ]

        system_lower = system_content.lower()
        has_role = any(indicator in system_lower for indicator in role_indicators)

        assert has_role, (
            "O system prompt deve definir uma persona/role para o modelo. "
            "Exemplo: 'Você é um Product Manager Sênior com 10 anos de experiência...'"
        )

    def test_prompt_mentions_format(self):
        """Verifica se o prompt exige formato Markdown ou User Story padrão."""
        prompt = load_prompt()
        system_content = get_system_content(prompt)
        system_lower = system_content.lower()

        format_indicators = [
            "como [",           # template "Como [persona]..."
            "como um",          # estrutura user story
            "eu quero",         # estrutura user story
            "para que",         # estrutura user story
            "critérios de aceitação",
            "dado que",         # formato gherkin
            "quando",
            "então",
            "user story",
            "markdown",
        ]

        matched = [ind for ind in format_indicators if ind in system_lower]

        assert len(matched) >= 3, (
            "O prompt deve mencionar explicitamente o formato esperado de saída "
            "(User Story padrão: Como / Eu quero / Para que, ou formato Gherkin). "
            f"Indicadores encontrados: {matched}"
        )

    def test_prompt_has_few_shot_examples(self):
        """Verifica se o prompt contém exemplos de entrada/saída (técnica Few-shot)."""
        prompt = load_prompt()
        system_content = get_system_content(prompt)
        system_lower = system_content.lower()

        # Verifica presença de marcadores de exemplos
        example_markers = ["exemplo", "relato:", "user story:"]
        has_markers = any(marker in system_lower for marker in example_markers)

        # Verifica se há pelo menos 2 exemplos (few-shot = múltiplos exemplos)
        example_count = system_lower.count("exemplo ")
        has_multiple = example_count >= 2

        # Verifica padrão de entrada/saída nos exemplos
        has_input_output = (
            "relato:" in system_lower or "relato de bug:" in system_lower
        ) and (
            "user story:" in system_lower or "user story gerada:" in system_lower
        )

        assert has_markers, (
            "O prompt deve conter exemplos Few-shot com marcadores como "
            "'Exemplo', 'Relato:', 'User Story:'"
        )
        assert has_multiple, (
            f"O prompt deve conter pelo menos 2 exemplos Few-shot. "
            f"Encontrados: ~{example_count}"
        )
        assert has_input_output, (
            "Os exemplos Few-shot devem mostrar pares de entrada (Relato) "
            "e saída (User Story gerada)."
        )

    def test_prompt_no_todos(self):
        """Garante que não há nenhum [TODO] esquecido no texto."""
        prompt = load_prompt()
        all_content = get_all_content(prompt)

        # Verifica também nos metadados e descrição
        full_text = yaml.dump(prompt, allow_unicode=True)

        todos_found = []
        for i, line in enumerate(full_text.splitlines(), 1):
            if "[TODO]" in line.upper() or "[todo]" in line.lower():
                todos_found.append(f"  Linha {i}: {line.strip()}")

        assert not todos_found, (
            "O prompt contém [TODO]s não resolvidos:\n" + "\n".join(todos_found)
        )

    def test_minimum_techniques(self):
        """Verifica via metadados do YAML se pelo menos 2 técnicas foram listadas."""
        prompt = load_prompt()

        metadata = prompt.get("metadata", {})
        assert metadata, (
            "O arquivo YAML deve conter uma seção 'metadata'."
        )

        techniques = metadata.get("techniques", [])
        assert isinstance(techniques, list), (
            "metadata.techniques deve ser uma lista. "
            f"Tipo encontrado: {type(techniques)}"
        )
        assert len(techniques) >= 2, (
            f"Devem ser listadas pelo menos 2 técnicas em metadata.techniques. "
            f"Encontradas: {len(techniques)} → {techniques}"
        )

        # Verifica que as técnicas não estão vazias
        non_empty = [t for t in techniques if t and str(t).strip()]
        assert len(non_empty) >= 2, (
            f"As técnicas listadas não podem ser strings vazias. "
            f"Técnicas válidas encontradas: {non_empty}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
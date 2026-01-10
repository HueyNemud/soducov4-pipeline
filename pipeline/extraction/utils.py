from pipeline.chunking.schemas import Chunk


def format_chunk_as_numbered_lines(chunk: Chunk) -> str:
    """
    Prepares a text chunk for LLM processing by prepending line numbers.
    Format example: '0 @ Content of first line'
    """
    return "\n".join(f"{index} @ {line.text}" for index, line in enumerate(chunk))

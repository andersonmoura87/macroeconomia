"""Erros tipados da ingestão BCB SGS (diagnóstico e tratamento em pipeline)."""


class BcbSgsError(RuntimeError):
    """Erro base da integração com o SGS."""


class BcbSgsNetworkError(BcbSgsError):
    """Falha de transporte (DNS, timeout, conexão resetada, TLS, etc.)."""


class BcbSgsHttpError(BcbSgsError):
    """Resposta HTTP não-sucesso após esgotar tentativas (quando aplicável)."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class BcbSgsPayloadError(BcbSgsError):
    """Corpo inesperado: JSON inválido, tipo errado ou linhas SGS corrompidas."""

"""Best-effort handoff notifications through a configured generic HTTPS webhook."""
import httpx

from .config import settings


async def notify_new_ticket(ticket_id: int, question: str) -> None:
    """Notify an automation endpoint; no URL means notifications are intentionally disabled."""
    if not settings.notification_webhook_url:
        return
    headers = {"Authorization": f"Bearer {settings.notification_webhook_token}"} if settings.notification_webhook_token else {}
    async with httpx.AsyncClient(timeout=settings.provider_timeout_seconds) as client:
        response = await client.post(settings.notification_webhook_url,
                                     json={"event": "ticket.created", "ticket_id": ticket_id, "question": question},
                                     headers=headers)
        response.raise_for_status()

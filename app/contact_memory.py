import json
import time
from pathlib import Path
from app.config import ROOT

def find_recent_interrupted_call(phone: str) -> dict | None:
    """Search logs/calls/ for a recent call dropped within the last 10 minutes."""
    clean_phone = ''.join(filter(str.isdigit, str(phone)))[-10:]
    if not clean_phone:
        return None
    
    log_dir = Path(ROOT) / "logs" / "calls"
    if not log_dir.exists():
        return None
        
    log_files = sorted(log_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    for path in log_files:
        try:
            # Ignore logs older than 10 minutes (600 seconds) so old tests don't trigger this
            if time.time() - path.stat().st_mtime > 600:
                continue
                
            data = json.loads(path.read_text(encoding="utf-8"))
            rec_phone = ''.join(filter(str.isdigit, str(data.get("phone", ""))))[-10:]
            
            if rec_phone == clean_phone:
                outcome = data.get("outcome", {})
                ended_by = outcome.get("ended_by", "")
                duration = data.get("duration_secs", 0)
                
                if ended_by in ("dropped", "timeout", "error", "network") and duration < 180:
                    return {
                        "transcript": data.get("transcript", []),
                        "data": outcome.get("data", {}),
                        "summary": outcome.get("summary", "")
                    }
        except Exception:
            continue
            
    return None
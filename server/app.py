import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import uvicorn
from fastapi import FastAPI
from environment import AIHumanEnv, Observation

app = FastAPI()
env = AIHumanEnv()

@app.post("/reset")
async def reset_env() -> dict:
    obs: Observation = env.reset()
    return {"status": "ok", "observation": obs.model_dump()}

@app.get("/state")
async def get_state() -> dict:
    obs = env.state()
    return {"observation": obs.model_dump() if obs else None}

def main():
    uvicorn.run(app, host="0.0.0.0", port=8080)

if __name__ == "__main__":
    main()

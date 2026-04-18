from pydantic import BaseModel, ConfigDict
from typing import Literal, Any


class FunctionParamDef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["string", "number", "boolean", "object", "array"]


class FunctionReturnsDef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["string", "number", "boolean", "object", "array"]


class FunctionDef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: dict[str, FunctionParamDef]
    returns: FunctionReturnsDef


class PromptDef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str


class FunctionCallOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str
    name: str
    parameters: dict[str, Any]

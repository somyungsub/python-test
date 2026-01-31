from pydantic import BaseModel, Field


class Recipe(BaseModel):
    ingredients: list[str] = Field(description="ingredients of dish")
    steps: list[str] = Field(description="steps to make dish")

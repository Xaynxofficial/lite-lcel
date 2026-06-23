import inspect
from typing import Any, Callable, Dict, List

class StructuredTool:
    """
    Python fonksiyonlarını şemalandıran ve yürütülebilir araçlara dönüştüren sınıf.
    """
    def __init__(self, name: str, description: str, args_schema: Dict[str, Any], func: Callable, required_args: List[str] = None):
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.func = func
        self.required_args = required_args or list(args_schema.keys())

    def invoke(self, args: Dict[str, Any]) -> Any:
        """Aracı verilen argümanlarla çalıştırır."""
        return self.func(**args)

    def to_openai_tool(self) -> Dict[str, Any]:
        """
        OpenAI API'sinin beklediği JSON fonksiyon şemasına dönüştürür.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.args_schema,
                    "required": self.required_args
                }
            }
        }

    def __call__(self, *args, **kwargs) -> Any:
        return self.func(*args, **kwargs)


def _get_json_type(py_type: Any) -> str:
    if py_type is int:
        return "integer"
    elif py_type is float:
        return "number"
    elif py_type is bool:
        return "boolean"
    elif py_type is str:
        return "string"
    return "string"


def tool(func: Callable) -> StructuredTool:
    """
    Python fonksiyonlarını otomatik olarak bir StructuredTool nesnesine dönüştüren dekoratör.
    Fonksiyon tip ipuçlarını (type hints) ve docstring'i okuyarak JSON şemasını otomatik çıkartır.
    """
    name = func.__name__
    doc = func.__doc__ or ""
    description = doc.strip().split("\n")[0] if doc else f"Python tool: {name}"
    
    # Parametreleri denetle
    sig = inspect.signature(func)
    args_schema = {}
    required_args = []
    
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
            
        # Parametre tipini çıkar
        py_type = param.annotation if param.annotation != inspect.Parameter.empty else str
        json_type = _get_json_type(py_type)
        
        args_schema[param_name] = {
            "type": json_type,
            "description": f"{param_name} parametresi"
        }
        
        # Varsayılan değeri yoksa zorunlu alan olarak işaretle
        if param.default == inspect.Parameter.empty:
            required_args.append(param_name)
            
    return StructuredTool(
        name=name,
        description=description,
        args_schema=args_schema,
        func=func,
        required_args=required_args
    )

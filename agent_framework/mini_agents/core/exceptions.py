"""调用方可以统一捕获 AgentError，也可以分别处理模型和流程错误。"""


class AgentError(RuntimeError):
    pass


class ModelError(AgentError):
    pass


class OutputError(AgentError):
    pass


class StepLimitError(AgentError):
    pass

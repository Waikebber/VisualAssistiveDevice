# Initialize configuration manager
from .core.config_manager import ConfigManager

# Initialize system state
from .core.system_state import SystemState

# Make configuration and state available at the package level
config_manager = ConfigManager()
system_state = SystemState()
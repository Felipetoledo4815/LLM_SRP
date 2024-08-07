import os
from dotenv import load_dotenv

init_abs_path = os.path.abspath(__file__)
module_dir = os.path.dirname(init_abs_path)

load_dotenv(os.path.join(module_dir, '.env'))

from slowapi import Limiter                        # type:ignore
from slowapi.util import get_remote_address   # type:ignore

# initialized to protect from circular import error
limiter = Limiter(key_func=get_remote_address)
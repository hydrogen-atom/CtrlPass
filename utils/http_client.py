import ssl
import urllib3
import requests
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager


class _CompatibleSSLAdapter(HTTPAdapter):
    """Custom adapter to handle TLS compatibility issues on some environments."""

    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        context = ssl.create_default_context()
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.set_ciphers("DEFAULT@SECLEVEL=1")
        pool_kwargs["ssl_context"] = context
        self.poolmanager = PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            **pool_kwargs,
        )


_session: requests.Session | None = None


def get_session() -> requests.Session:
    """Return a requests.Session with custom TLS adapter mounted."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.mount("https://", _CompatibleSSLAdapter())
    return _session

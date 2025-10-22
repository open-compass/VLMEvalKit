# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import os

from math_verify.errors import TimeoutException

TIMEOUT_WARNING_SHOWN = False
logger = logging.getLogger(__name__)


def timeout(timeout_seconds: int | None = 10):  # noqa: C901
    """A decorator that applies a timeout to the decorated function.

    Args:
        timeout_seconds (int): Number of seconds before timing out the decorated function.
                 Defaults to 10 seconds.

    Notes:
        On Unix systems, uses a signal-based alarm approach which is more efficient as it doesn't require spawning a new process.
        On Windows systems, uses a multiprocessing-based approach since signal.alarm is not available. This will incur a huge performance penalty.
    """
    if timeout_seconds is None or timeout_seconds <= 0:

        def no_timeout_decorator(func):
            return func

        return no_timeout_decorator

    if os.name == "posix":
        # Unix-like approach: signal.alarm
        import signal

        def decorator(func):
            def handler(signum, frame):
                raise TimeoutException("Operation timed out!")

            def wrapper(*args, **kwargs):
                old_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(timeout_seconds)
                try:
                    return func(*args, **kwargs)
                finally:
                    # Cancel the alarm and restore previous handler
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

            return wrapper

        return decorator

    else:
        # Windows approach: use multiprocessing
        from multiprocessing import Process, Queue

        def decorator(func):
            def wrapper(*args, **kwargs):
                q = Queue()

                def run_func(q, args, kwargs):
                    try:
                        result = func(*args, **kwargs)
                        q.put((True, result))
                    except Exception as e:
                        q.put((False, e))

                p = Process(target=run_func, args=(q, args, kwargs))
                p.start()
                p.join(timeout_seconds)

                if p.is_alive():
                    # Timeout: Terminate the process
                    p.terminate()
                    p.join()
                    raise TimeoutException("Operation timed out!")

                # If we got here, the process completed in time.
                success, value = q.get()
                if success:
                    return value
                else:
                    # The child raised an exception; re-raise it here
                    raise value

            return wrapper

        return decorator

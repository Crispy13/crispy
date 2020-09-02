# Crispy
Functions for personal use

## Default Logger
This module have a default logger, you can add a file handler to the logger as the following:

```
import crispy as csp

csp.logger.addHandler(csp.make_file_handler("path/log_file.log"))
```

## Play audio when exceptions are occurred
```
get_ipython().set_custom_exc((Exception,), exception_sound(logger)) # Exception sound alert
```

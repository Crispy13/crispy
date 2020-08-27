# Crispy13
Functions for personal use

This module have a default logger, you can add a file handler to the logger as the following:

```
import crispy as csp

csp.logger.addHandler(csp.make_file_handler("path/log_file.log"))
```

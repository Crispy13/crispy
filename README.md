# Crispy
Functions for personal use


## Examples
Notebook examples using this repo are in /examples folder.

- Example list :
  * [Guided Grad-CAM with Tensorflow 2][ggc]  





<hr>
## Default Logger
This module have a default logger, you can add a file handler to the logger as the following:

```
import crispy as csp
csp.logger.addHandler(csp.make_file_handler("path/log_file.log"))
```


## Play audio when exceptions are occurred
Tested in jupyter lab
```
get_ipython().set_custom_exc((Exception,), csp.exception_sound(csp.logger)) # Exception sound alert
```

[ggc]: https://github.com/Crispy13/crispy/blob/master/examples/guided_grad_cam.ipynb

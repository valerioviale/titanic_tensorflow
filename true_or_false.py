# 1. When we set `a = 256` and `b = 256`, they both refer to the same memory 
# location because 256 falls within the range of cached small integers. Therefore, 
#`a is b` returns `True`.

a = 256
b = 256
a is b
 

# 2. When we set `a = 257` and `b = 257`, they are no longer within the range of cached 
# small integers, so they are not interned, and `a is b` returns `False`.


a = 257
b = 257
a is b
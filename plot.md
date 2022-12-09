![image](https://user-images.githubusercontent.com/23145731/206644509-2f67f682-b0e9-4a9a-9a46-54c675c79b14.png)


```python
plt.subplots(1, 1)
x= range(100)
y= [i**2 for i in x]
 
plt.plot(x, y, linewidth = '1', label = "test", color=' coral ', linestyle=':', marker='|')
plt.legend(loc='upper left')
plt.show()
```

linestyle可选参数：
```
'-'       solid line style
'--'      dashed line style
'-.'      dash-dot line style
':'       dotted line style
```

marker可选参数：

```
'.'       point marker
','       pixel marker
'o'       circle marker
'v'       triangle_down marker
'^'       triangle_up marker
'<'       triangle_left marker
'>'       triangle_right marker
'1'       tri_down marker
'2'       tri_up marker
'3'       tri_left marker
'4'       tri_right marker
's'       square marker
'p'       pentagon marker
'*'       star marker
'h'       hexagon1 marker
'H'       hexagon2 marker
'+'       plus marker
'x'       x marker
'D'       diamond marker
'd'       thin_diamond marker
'|'       vline marker
'_'       hline marker
```

![image](https://user-images.githubusercontent.com/23145731/206644912-620fc21a-7607-436b-bb59-2841943302ea.png)



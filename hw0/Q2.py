from PIL import Image
file = input()
im = Image.open(file)
pixels = list(im.getdata())
out_list = [list(t) for t in pixels]
new = []
for inside_list in out_list:
    haha = []
    for number in inside_list:
        haha.append(number//2)
    new.append(haha)
this = [tuple(i) for i in new]
im2 = Image.new(im.mode, im.size)
im2.putdata(this)
im2.save('Q2.png')

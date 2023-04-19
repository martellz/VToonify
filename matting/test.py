from rembg_simplify import remove
import cv2

input_path = '../data/077436.jpg'
output_path = '../data/077436_removal.jpg'

# with open(input_path, 'rb') as i:
#   with open(output_path, 'wb') as o:
#     input = i.read()
#     output = remove(input)
#     print(output)
#     o.write(output)

img = cv2.imread(input_path)
output = remove(img)
print(output.max(), output.dtype)
cv2.imwrite(output_path, output)
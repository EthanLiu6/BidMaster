import os
import subprocess

source = input('请输入需要转换的文件路径：')
dest = input('请输入输出路径（若不存在该路径会报错，不会新建）：')
g = os.listdir(source)
file_path = [f for f in g if f.endswith(('.doc'))]
print(file_path)
for i in file_path:
    file = (source + '/' + i)
    print(file)
    output = subprocess.check_output(
        ["/Applications/LibreOffice.app/Contents/MacOS/soffice", "--headless", "--convert-to", "docx", file, "--outdir",
         dest])
print('success!')

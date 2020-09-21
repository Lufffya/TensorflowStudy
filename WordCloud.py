#
# 提取聚类中的词频关键字
# 使用 wordCloud 第三方包
# GitHub：https://github.com/amueller/word_cloud/
# 官方文档：http://amueller.github.io/word_cloud/
#

import xlrd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 打开文件
review_Data = xlrd.open_workbook('DataSet\\ReviewKMaensResult.xlsx')

# 查看工作表
print("工作表：" + str(review_Data.sheet_names()))

# 遍历所有Sheet
for sheet in review_Data.sheets():
    print("============================")
    print("Sheet名称：" + str(sheet.name))
    print("总行数：" + str(sheet.nrows))
    print("总列数：" + str(sheet.ncols))
    colReviews = sheet.col_values(0)
    textReviews = " ".join(colReviews)

    # 生成词云图像
    # max_words：最大字数
    wordcloud = WordCloud(max_words=10).generate(textReviews)
    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


# 通过文件名获得工作表,获取工作表
# sheet = review_Data.sheet_by_name('聚类0')
# sheet = review_Data.sheet_by_index(0)

# 获取整行的值 和整列的值，返回的结果为数组
# 整行值：sheet.row_values(start,end)
# 整列值：sheet.col_values(start,end)
# 参数 start 为从第几个开始打印，
# end为打印到那个位置结束，默认为none
# print("整行值：" + str(sheet.row_values(0)))
# print("整列值：" + str(sheet.col_values(0)))

# 获取某个单元格的值，例如获取B3单元格值
# cel_B3 = sheet.cell(0,0).value
# print("第三行第二列的值：" + cel_B3)

# 读取词库数据集
# text = open("DataSet\\TestWordCloud.txt").read()

# The pil way (if you don't have matplotlib)
# image = wordcloud.to_image()
# image.show()
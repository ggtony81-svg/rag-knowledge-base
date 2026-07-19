import pymysql
try:
    conn = pymysql.connect(host='localhost', user='root', password='123456')
    print('MySQL 连接成功 ✅')
    conn.close()
except Exception as e:
    print(f'连接失败: {e}')

import requests
import csv
 
f = open('comments.csv',mode='a',encoding='utf-8-sig',newline='')
csv_write = csv.writer(f)
csv_write.writerow(['id','screen_name','text_raw','like_counts','total_number','created_at'])
headers = {
        'cookie': '_T_WM=1abb4ffbd2abf8c60ec2d6b8ff6b2bc5; WEIBOCN_FROM=1110006030; SCF=ArqeQWRs0ShEQmCXKVZM_KLKNMXKzFbtNwG0DjQW6zBimyrPmiWswD33UE5wcZIoBrXfntFVXwOHT9URcwIlG6s.; SUB=_2A25FDj1VDeRhGeFH4lEW-C7OwjuIHXVmYjCdrDV6PUJbktANLUv6kW1NehXBx2PDlIE5__xqAkWPpK7M3LC38nYY; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WhoN5PmMwvk.5X25w5cir8x5JpX5KMhUgL.FoM41KeN1h5E1KM2dJLoIp7LxKML1KBLBKnLxKqL1hnLBoMN1K.0S0n7eo.N; SSOLoginState=1745505541; ALF=1748097541; MLOGIN=1; XSRF-TOKEN=d75a89; mweibo_short_token=4271f6fb3d; M_WEIBOCN_PARAMS=luicode%3D10000011%26lfid%3D102803%26uicode%3D20000174',
        'referer': 'https://weibo.com/7190522839/O1kt4jTyM',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.5359.95 Safari/537.36'
    }
 
 
def get_next(next='count=10'):
    url = f'https://weibo.com/ajax/statuses/buildComments?is_reload=1&id=5157139102827239&is_show_bulletin=2&is_mix=0&{next}&uid=7190522839&fetch_level=0&locale=zh-CN'
 
    response = requests.get(url=url,headers=headers)
    json_data = response.json()
 
    data_list = json_data['data']
    max_id = json_data['max_id']
    for data in data_list:
        text_raw = data['text_raw']
        id = data['id']
        created_at = data['created_at']
        like_counts = data['like_counts']
        total_number = data['total_number']
        screen_name = data['user']['screen_name']
        print(id,screen_name,text_raw,like_counts,total_number,created_at)
 
        csv_write.writerow([id,screen_name,text_raw,like_counts,total_number,created_at])
 
    max_str = 'max_id='+str(max_id)
    get_next(max_str)
get_next()
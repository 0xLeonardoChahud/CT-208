import requests

url = "https://dkmgames.com/Suguru/SuguruServer.php"
ua='User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

headers = {'User-Agent':ua}

sizes = [5,6,7,8]
difficulties = [0,1,2]

for size in sizes:
    for diff in difficulties:
        r = requests.post(url, headers=headers, data={'action':'getpuzzle', 'level':diff, 'size':size, 'id':0})
        





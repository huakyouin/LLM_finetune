## Note selenium>4.6, otherwise need to install webdirver yourself
## used for collecting stock news

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from datetime import datetime
from pathos.multiprocessing import ProcessPool
import pandas as pd
import os
import numpy as np

# 设置无头模式
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument('--log-level=3')


def get_news_details(driver,url):
    driver.get(url)
    title = driver.find_element(By.CSS_SELECTOR,".newstitle").text.strip()
    time = driver.find_element(By.CSS_SELECTOR,".newsauthor .time").text.strip()
    # content = " ".join([p.text.strip() for p in driver.find_elements(By.CSS_SELECTOR,".newstext p")])
    content = driver.find_element(By.CSS_SELECTOR,".newstext div").text.replace('\n', ' ')
    

    return title,content,time


def crwal_one_stock(stock_id,start_date,end_date,base_url = "https://guba.eastmoney.com/list,{0},1,f_{1}.html"):
    driver = webdriver.Chrome(options=chrome_options)
    first_message_time = datetime.now()
    pages = 1
    tot_items = []

    while first_message_time > start_date:
        driver.get(base_url.format(stock_id,pages))
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'table_list')))
        hrefitems = [item.get_attribute("href") for item in driver.find_elements(By.CSS_SELECTOR, '.listitem .title a')]
        if len(hrefitems)==0:
            break
        _,_,first_message_time = get_news_details(driver,hrefitems[-1])
        first_message_time = datetime.strptime(first_message_time, '%Y-%m-%d %H:%M:%S')

        if first_message_time <= end_date:
            tot_items += hrefitems
        pages += 1
        

    res = []
    for href in tot_items:
        title,content,time = get_news_details(driver,href)
        if start_date<= datetime.strptime(time, '%Y-%m-%d %H:%M:%S') <= end_date:
            res.append((time, title, content))
        if datetime.strptime(time, '%Y-%m-%d %H:%M:%S') < start_date:
            break

    # 关闭浏览器驱动
    driver.quit()
    return res

def crawl_stocks_and_save(news_base,start_date,end_date,stock_id_list):
    for stock_id in stock_id_list:
        news_path = os.path.join(news_base, f"{stock_id}.csv")
        
        if os.path.exists(news_path):
            print("! 股票 {} 已采集过".format(stock_id))
            continue
        print("股票 {} 采集中".format(stock_id))
        res = crwal_one_stock(stock_id,start_date,end_date)
        print("股票 {} 采集完成，共{}条记录。".format(stock_id,len(res)))

        # 创建 DataFrame
        df = pd.DataFrame(res, columns=["Time", "Title", "Content"])
        df.to_csv(news_path, index=False, quoting=1)  # quoting=1 等同于 csv.QUOTE_ALL


if __name__=="__main__":
    start_date, end_date = datetime(2024, 6, 1), datetime(2024, 6, 15)
    stock_codes = ["000001","000002","000003","000004"]
    news_base = "stock_news"
    os.makedirs(news_base,exist_ok=True)
    NUM_PROCS = 2
    pool = ProcessPool(NUM_PROCS)
    split_codes = np.array_split(stock_codes, NUM_PROCS)
    pool.map(crawl_stocks_and_save,[news_base]*NUM_PROCS, [start_date]*NUM_PROCS, [end_date]*NUM_PROCS, split_codes)
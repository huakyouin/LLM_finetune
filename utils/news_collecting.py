# Note: selenium>4.6, or you need to install webdriver on your own
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from pathos.multiprocessing import ProcessPool
from datetime import datetime
import os
import re
import pandas as pd
import numpy as np


# 设置无头模式
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")


def get_news_details(driver,url):
    driver.get(url)
    title = driver.find_element(By.CSS_SELECTOR,".newstitle").text.strip()
    time = driver.find_element(By.CSS_SELECTOR,".newsauthor .time").text.strip()
    content = " ".join([p.text.strip() for p in driver.find_elements(By.CSS_SELECTOR,".newstext p")])
    return title,content,time


def crwal_one_stock(stock_id, base_url = "https://guba.eastmoney.com/list,{0},1,f_{1}.html"):
    global start_date, end_date
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
            res.append(time,title,content)
        if datetime.strptime(time, '%Y-%m-%d %H:%M:%S') < start_date:
            break

    # 关闭浏览器驱动
    driver.quit()
    return res

def crawl_stocks_and_save(stock_id_list):
    for stock_id in stock_id_list:
        news_path = os.path.join(news_base, f"{stock_id}.csv")
        
        if os.path.exists(news_path):
            print("! 股票 {} 已采集过".format(stock_id))
            continue

        res = crwal_one_stock(stock_id)

        # 创建 DataFrame
        df = pd.DataFrame(res, columns=["Time", "Title", "Content"])
        df.to_csv(news_path, index=False, quoting=1)  # quoting=1 等同于 csv.QUOTE_ALL


if __name__=="__main__":
    start_date, end_date = datetime(2017, 1, 1), datetime(2020, 1, 1)
    stock_codes = ["000001"]
    news_base = "data/stock_news"
    os.makedirs(news_base, exist_ok=True)
    NUM_PROCS = 10
    pool = ProcessPool(NUM_PROCS)
    split_codes = np.array_split(stock_codes, NUM_PROCS)
    pool.map(crawl_stocks_and_save, split_codes)


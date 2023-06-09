from utility.amazon.amazon_review_scrapper import scrapped
from utility.amazon.amazon_senti import senti
import pandas as pd
import time

def amazonMain(productUrl, uuid):
    reviewList = []
    flag = 0
    for i in range(10):
        print(f"Running for page {i}")
        try: 
            reviewUrl = productUrl + '&pageNumber=' + str(i)
            list = scrapped(reviewUrl)
            if list is not None:
                print(f'{len(list)} ====  {flag}')
                reviewList.extend(list)
            else:
                print(flag)
                flag = flag + 1

            if(flag >= 5):
                print(f"No reviews found on page {i}")
                break

        except Exception as e:
            print(e)
            flag = flag + 1
            print(flag)

            if(flag >= 5):
                print(f"No reviews found on page {i}")
                break
        
        time.sleep(10)

    output_df = pd.DataFrame.from_dict(reviewList)
    # output_df.to_csv(f'data/{uuid}.csv', index=False, header=["title","content","date","variant","images","verified","author","rating","product","url"])
    senti('amazon-train')
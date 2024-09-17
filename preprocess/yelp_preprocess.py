import pickle
import gzip
import json
import pprint
import argparse
import os
from datetime import datetime
import logging
import pandas as pd
import random

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logging.info('Admin logged in')

pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint
DATE_PATTERN = "%Y-%m-%d %H:%M:%S"

def get_created_time(text):
    return int(datetime.strptime(text, DATE_PATTERN).timestamp())

def convert_data(input_file, output_file, restaurant_business_ids):
    try:
        if os.path.exists(output_file) and not args.force:
            logging.info(f"Grouped reviews file already exists at {output_file}. Skipping conversion.")
            return

        with open(input_file) as fin, gzip.open(output_file, "wt") as fout:
            business_dict = {}
            count = 0
            for line in fin:
                data = json.loads(line)
                business_id = data["business_id"]
                if business_id not in restaurant_business_ids:
                    continue
                if business_id not in business_dict:
                    business_dict[business_id] = []
                count += 1
                business_dict[business_id].append((
                    get_created_time(data["date"]),
                    data["user_id"],
                    data["text"],
                    data["stars"]))
                if count % 100000 == 0:
                    logging.info(f"Processed {count} reviews")
            
            logging.info(f"Total businesses with reviews: {len(business_dict)}")
            for b in business_dict:
                business_dict[b].sort()
                fout.write("%s\n" % json.dumps({"business": b, "reviews": business_dict[b]}))
        
        logging.info(f"Grouped reviews file created at {output_file}")
    except Exception as e:
        logging.error(f"Error in convert_data: {str(e)}")
        raise

def process_reviews(input_file, output_file, restaurant_business_ids, num_review):
    try:
        if os.path.exists(output_file) and not args.force:
            logging.info(f"Processed reviews file already exists at {output_file}. Skipping processing.")
            return set()

        processed_businesses = set()
        with gzip.open(input_file, 'rt') as f_in, gzip.open(output_file, "wt") as fout:
            for l in f_in:
                r = json.loads(l)
                if r['business'] not in restaurant_business_ids:
                    continue
                if len(r['reviews']) >= num_review:
                    tmp = {"reviews":[], "scores":[]}
                    tmp['business'] = r['business']
                    processed_businesses.add(r['business'])
                    for i in range(min(10, len(r['reviews']))):
                        tmp["reviews"].append(r["reviews"][i][2])
                        tmp["scores"].append(r["reviews"][i][3])
                    tmp["avg_score"] = sum([r["reviews"][j][3] for j in range(min(num_review, len(r['reviews'])))])/min(num_review, len(r['reviews']))
                    fout.write("%s\n" % json.dumps(tmp))
        
        logging.info(f"Total businesses processed: {len(processed_businesses)}")
        return processed_businesses
    except Exception as e:
        logging.error(f"Error in process_reviews: {str(e)}")
        raise

def check_file_content(file_path):
    try:
        with gzip.open(file_path, 'rt') as f:
            content = f.read().strip()
            if content:
                logging.info(f"File {file_path} is not empty. First few characters: {content[:50]}...")
            else:
                logging.warning(f"File {file_path} is empty.")
    except Exception as e:
        logging.error(f"Error checking file {file_path}: {str(e)}")

def compare_business_ids(processed_ids, split_ids):
    logging.info("Comparing business IDs:")
    logging.info(f"Processed businesses: {len(processed_ids)}")
    logging.info(f"Split businesses: {len(split_ids)}")
    common_ids = processed_ids.intersection(split_ids)
    logging.info(f"Common business IDs: {len(common_ids)}")
    
    if len(common_ids) == 0:
        logging.warning("No common business IDs found!")
        logging.info("Sample of processed business IDs:")
        for bid in random.sample(list(processed_ids), min(5, len(processed_ids))):
            logging.info(bid)
        logging.info("Sample of split business IDs:")
        for bid in random.sample(list(split_ids), min(5, len(split_ids))):
            logging.info(bid)
    else:
        logging.info("Sample of common business IDs:")
        for bid in random.sample(list(common_ids), min(5, len(common_ids))):
            logging.info(bid)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--yelp_data_dir", default="/data/yelp", type=str, help="")
    parser.add_argument("--output_dir", default="/data/joe/yelp", type=str, help="")
    parser.add_argument("--data_split", action='store_true', default=True, help="")
    parser.add_argument("--num_review", type=int, default=50, help="Number of reviews for computing average rating")    
    parser.add_argument("--force", action='store_true', help="Force reprocessing even if files exist")
    
    args = parser.parse_args() 
    logging.info(f"args: {args}")   

    try:
        restaurant_business_ids = set()
        business_file = os.path.join(args.yelp_data_dir, "business.json")
        if not os.path.exists(business_file):
            raise FileNotFoundError(f"business.json not found in {args.yelp_data_dir}")

        with open(business_file, "r") as f:    
            for line in f:  
                if line:    
                    json_content = json.loads(line)
                    if json_content["categories"] is not None:                        
                        categories = [val.lower().strip() for val in json_content["categories"].split(",")]
                        if "restaurants" in categories:
                            restaurant_business_ids.add(json_content["business_id"])
        
        logging.info(f"Finished reading the business.json file. There are {len(restaurant_business_ids)} unique restaurants in the dataset.")
        logging.info("Sample of restaurant business IDs:")
        for bid in random.sample(list(restaurant_business_ids), min(5, len(restaurant_business_ids))):
            logging.info(bid)

        grouped_reviews_filepath = os.path.join(args.yelp_data_dir, "grouped_reviews.jsonlist.gz")
        if args.force or not os.path.exists(grouped_reviews_filepath):
            logging.info(f'Building grouped reviews at {grouped_reviews_filepath}')
            convert_data(os.path.join(args.yelp_data_dir, "review.json"), grouped_reviews_filepath, restaurant_business_ids)
        else:
            logging.info(f'Grouped reviews file already exists at {grouped_reviews_filepath}')
        
        check_file_content(grouped_reviews_filepath)

        logging.info(f"Converting grouped reviews into restaurant only reviews and compute average of first {args.num_review} reviews")
        out_file = os.path.join(args.output_dir, f"yelp_10reviews_{args.num_review}avg.jsonl.gz")
        processed_businesses = process_reviews(grouped_reviews_filepath, out_file, restaurant_business_ids, args.num_review)
        
        check_file_content(out_file)

        if args.data_split:
            split_datapath = os.path.join(args.output_dir, f"{args.num_review}reviews")
            os.makedirs(split_datapath, exist_ok=True)
            
            def dump_jsonl_gz(obj, outpath):
                with gzip.open(outpath, "wt") as fout:
                    for o in obj:
                        fout.write("%s\n" % json.dumps(o))
            
            splits = ["train", "dev", "test"]
            split_ids = {}
            all_split_ids = set()
            for s in splits:
                csv_path = os.path.join(os.path.dirname(__file__), f"{s}_business_ids.csv")
                if not os.path.exists(csv_path):
                    logging.error(f"CSV file not found: {csv_path}")
                    continue
                split_ids[s] = set(pd.read_csv(csv_path).business.values)
                all_split_ids.update(split_ids[s])
                logging.info(f"Loaded {len(split_ids[s])} business IDs for {s} split from {csv_path}")
            
            compare_business_ids(processed_businesses, all_split_ids)
            
            reviews = []
            with gzip.open(out_file, 'rt') as f:
                for line in f:
                    reviews.append(json.loads(line))
            
            logging.info(f"Total reviews loaded: {len(reviews)}")
            
            for split, ids in split_ids.items():
                split_reviews = []
                for review in reviews:
                    if review['business'] in ids:
                        split_reviews.append(review)

                storepath = os.path.join(split_datapath, f"{split}.jsonl.gz")
                dump_jsonl_gz(split_reviews, storepath)
                logging.info(f"Data split length: {split} ({len(split_reviews)}), stored at: {storepath}")
                check_file_content(storepath)

            #Debug: Check for mismatched business IDs
            all_business_ids = set(review['business'] for review in reviews)
            for split, ids in split_ids.items():
                missing_ids = ids - all_business_ids
                if missing_ids:
                    logging.warning(f"{len(missing_ids)} business IDs in {split} split not found in processed data")
                    logging.warning(f"Sample of missing IDs: {list(missing_ids)[:5]}")
                
                extra_ids = all_business_ids - ids
                if extra_ids:
                    logging.warning(f"{len(extra_ids)} business IDs in processed data not found in {split} split")
                    logging.warning(f"Sample of extra IDs: {list(extra_ids)[:5]}")

            # debug
            logging.info(f"Total unique businesses in processed data: {len(all_business_ids)}")
            logging.info(f"Total unique businesses in split files: {len(set().union(*split_ids.values()))}")
            common_ids = all_business_ids.intersection(set().union(*split_ids.values()))
            logging.info(f"Number of common business IDs: {len(common_ids)}")

        logging.info("Finished!")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
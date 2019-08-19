from pymongo import MongoClient


# This function was for data selection from the whole dataSet
def collection_creator():
    client = MongoClient('mongodb://localhost:27017')
    sample_db = client.sample_dump
    db = client.dump
    sample_collection = sample_db.sample_issue_comments
    collection = db.issue_comments
    sample_collection.remove()

    cursor = collection.find({})

    # The better and more accurate way for getting part of data Was random selection,
    # but random algorithms could take time, so I had to select them in
    # this fast and good enough way.
    for index, item in enumerate(cursor):
        if index % 5 == 0:
            sample_collection.insert_one(item)
            print(" ======= {} === added ======".format(index))


if __name__ == "__main__":
    collection_creator()

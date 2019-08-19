import sys

import xlsxwriter
from pymongo import MongoClient

from tokenizer import text_processing


def writ_to_excel(file_name_path, data):
    workbook = xlsxwriter.Workbook(file_name_path + '.xlsx')
    worksheet = workbook.add_worksheet("statics")
    worksheet.set_column('A:A', 64)
    worksheet.merge_range(0, 0, 0, 1, data="some sample statistics of the data-set",
                          cell_format=workbook.add_format({
                              'align': 'center', 'bold': 1,
                              'border': 1, }))

    row = 1
    for key, value in data.items():
        row += 1
        worksheet.write(row, 0, key)
        worksheet.write(row, 1, value)

    workbook.close()


def issue_comments_extraction(d_type="word"):
    client = MongoClient('mongodb://localhost:27017')
    db = client.sample_dump
    collection = db.sample_issue_comments
    issues_ids = collection.distinct("issue_id")
    issues_number = len(issues_ids)  # number of issues ***********************
    average_comments_per_issue = collection.count_documents({}) / issues_number
    sum_of_involved_per_issues = 0

    for index, issue_id in enumerate(issues_ids):
        issue_id = 1
        print("===== {} of {} =====".format(index, len(issues_ids)))

        query = {"issue_id": issue_id}
        projection = {"body": u"", "_id": 0.0, "user": 1, "issue_id": 1}
        cursor = list(collection.find(query, projection=projection))
        sum_of_involved_per_issues += len(set(comment['user']['id'] for comment in cursor))

        if d_type == "word":
            words_extraction(cursor, issue_id)
        elif d_type == "text":
            text_extraction(cursor, issue_id)
        break
    average_involved = sum_of_involved_per_issues / len(issues_ids)
    statics = {"number of issues": issues_number,
               "average number of comments in one issue": average_comments_per_issue,
               "average number of people involved in the comments of one issue": average_involved}
    writ_to_excel("reports\\statics", statics)

    print(statics)
    print("\n========****  done  ****==========")


def words_extraction(cursor, issue_id):
    issue_comments_bodies = []
    for comment in cursor:
        comment_words = text_processing(comment['body'])
        if not (comment_words.count("request") or comment_words.count(
                "pull") or comment_words.count("pullrequest")):
            issue_comments_bodies += comment_words
    file = open('topic_modeling/issues_words/' + str(issue_id) + '.txt', 'w', encoding="utf-8")
    file.write(str(issue_comments_bodies))
    file.close()


def text_extraction(cursor, issue_id):
    issue_comments_bodies = []
    for comment in cursor:
        comment_body = comment['body']
        if not (comment_body.count("request") or comment_body.count("pull")):
            issue_comments_bodies.append(comment_body)
    file = open('clustering/issues_comments/' + str(issue_id) + '.txt', 'w', encoding="utf-8")
    file.write(str(issue_comments_bodies))
    file.close()


if __name__ == "__main__":
    d_type = sys.argv[1] if len(sys.argv) > 1 else "word"
    issue_comments_extraction(d_type)

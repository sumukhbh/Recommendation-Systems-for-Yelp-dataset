from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import sys
import time
import math
sc = SparkContext("local[*]", "Sumukh_Task2")
sc.setLogLevel("OFF")
start = time.time()
case = int(sys.argv[3])
input_file_path = sys.argv[1]
validation_file_path = sys.argv[2]
training_RDD = sc.textFile(input_file_path)
# saving the header #
header = training_RDD.first()
# filtering based on header #
training_data = training_RDD.filter(lambda a: a != header).map(lambda a: a.split(','))
#print(training_data.take(5))
user_RDD = training_data.map(lambda a: a[0]).distinct()
user_ids = user_RDD.collect()
#print(len(user_ids))
#print(user_ids)
business_RDD = training_data.map(lambda a: a[1]).distinct()
business_ids = business_RDD.collect()
#print(len(business_ids))
#print(business_ids)
# creating a dictionary for user ids #
user_count = list(range(len(user_ids)))
user_dict = dict(zip(user_ids,user_count))
inverse_user_mapping = dict(zip(user_count,user_ids))
#print(user_dict)
#print(inverse_user_mapping)
# creating a dictionary for business ids #
business_count = list(range(len(business_ids)))
business_dict = dict(zip(business_ids,business_count))
inverse_business_mapping = dict(zip(business_count,business_ids))
#print(business_dict)
#print(inverse_business_mapping)
# Computing average rating for  missing user_id #
user_average = dict()
missing_userid = training_data.map(lambda a: (user_dict[a[0]], [float(a[2])])).reduceByKey(lambda a,b: a+b).map(lambda a: (a[0], sum(a[1])/len(a[1]))).collect()
for id in missing_userid:
    user_average[id[0]] = id[1]
#print(user_average)
# Computing average rating for  missing business_id #
business_average = dict()
missing_businessid = training_data.map(lambda a: (business_dict[a[1]], [float(a[2])])).reduceByKey(lambda a,b: a+b).map(lambda a: (a[0], sum(a[1])/len(a[1]))).collect()
for id in missing_businessid:
    business_average[id[0]] = id[1]
#print(business_average)

validation_RDD = sc.textFile(validation_file_path).map(lambda a: a.split(","))
# saving the header #
header_1 = validation_RDD.first()
# filtering based on header #
validation_data = validation_RDD.filter(lambda a: a != header_1)
#print(validation_data.count())

# Model Based Collaborative Filtering #
if case == 1:
    ratings = training_data.map(lambda a: Rating(int(user_dict[a[0]]), int(business_dict[a[1]]), float(a[2])))
    rank = 8
    numIterations = 10
    model = ALS.train(ratings, rank, numIterations,0.2)
    #checking the existence of ids from validation set in training set #
    def check_id(id):
        if id[0] not in user_dict:
            user_id = -1
        elif id[0] in user_dict:
            user_id = user_dict[id[0]]
        if id[1] not in business_dict:
            business_id = -1
        elif id[1] in business_dict:
            business_id = business_dict[id[1]]
        return ((user_id, business_id))
    existing_data = validation_data.map(check_id)
    predictions = model.predictAll(existing_data).map(lambda r: ((r[0], r[1]), r[2]))
    # filtering the non existent ids #
    def filter_id(id):
        if id[0] not in user_dict or id[1] not in business_dict:
            return 1
    non_existing_data = validation_data.filter(filter_id).map(lambda a: (a[0], a[1]))
    # Handling Cold Start Problem #
    def predict_rating(id):
     if id[0] not in user_dict:
         rating = business_average[business_dict[id[1]]]
     elif id[1] not in business_dict:
         rating = user_average[user_dict[id[0]]]
     elif id[0] not in user_dict and id[1] not in business_dict:
         rating = 2.75
     return ((id[0], id[1]), rating)
    non_existing_data_prediction = non_existing_data.map(predict_rating).collect()
    #print(non_existing_data_prediction)
    existing_data_prediction = predictions.map(lambda a: ((inverse_user_mapping[a[0][0]], inverse_business_mapping[a[0][1]]), a[1])).collect()
    #print(existing_data_prediction)
    opened_file = open(sys.argv[4], "a+")
    opened_file.write("user_id, business_id, prediction\n")
    for id in existing_data_prediction:
        val = id[0][0] + "," + id[0][1] + "," + str(id[1]) + "\n"
        opened_file.write(val)
    for id in non_existing_data_prediction:
        val = id[0][0] + "," + id[0][1] + "," + str(id[1]) + "\n"
        opened_file.write(val)
    opened_file.close()

# For User Based and Item Based Collaborative Filtering #
ui_data = training_data.map(lambda a: ((user_dict[a[0]], business_dict[a[1]]), float(a[2])))
ui_result = ui_data.collect()
#print(ui_result)
ui_dict = dict()
for val in ui_result:
        if val[0][1] in ui_dict:
            ui_dict[val[0][1]][val[0][0]] = val[1]
        else:
            ui_dict[val[0][1]] = {val[0][0]:val[1]} # Stored as set to do set operations later #
#print(ui_dict)
new_validation_data = validation_data.map(lambda a: (a[0],a[1]))

# User Based Collaborative Filtering #
if case == 2:
    user_to_business = ui_data.map(lambda a: (a[0][0], {a[0][1]})).reduceByKey(lambda a, b: a.union(b)).collect()
    ub_dict = dict()
    for val in user_to_business:
        ub_dict[val[0]] = val[1]
    #print(ub_dict)

   # Pearson's correlation for User based CF #
    def Pearson_Correlation(val1, val2):
        co_rated = ub_dict[val1].intersection(ub_dict[val2])
        co_rated_length = len(co_rated)
        if co_rated_length == 0:
            return 0
        user1_weights = []
        user2_weights = []
        for x in co_rated:
            user1_weights.append(ui_dict[x][val1])
            user2_weights.append(ui_dict[x][val2])
        user1 = len(user1_weights)
        user2 = len(user2_weights)
        user1_avg = sum(user1_weights) / user1
        user2_avg = sum(user2_weights) / user2
        denom1 = 0
        denom2 = 0
        nume = 0
        user1_count = len(user1_weights)
        for x in range(0,user1_count):
            nume1 = user1_weights[x] - user1_avg
            nume2 = user2_weights[x] - user2_avg
            nume  = nume + nume1 * nume2
            denom1 = denom1 + nume1 ** 2
            denom2 = denom2 + nume2 ** 2
        denom1 = math.sqrt(denom1)
        denom2 = math.sqrt(denom2)
        denom = denom1 * denom2
        if nume == 0:
            return 0
        final_weight = nume/denom
        return final_weight

    similar_pairs = {}
    userid_count = len(inverse_user_mapping)
    for val1 in range(0,userid_count):
        similar_pairs[val1] = list()
        for val2 in range(val1 + 1, userid_count):
            similar = Pearson_Correlation(val1,val2)
            if similar != 0:
                sim_count = 0
                similar_pairs[val1].append((val2, similar))
                if val2 in similar_pairs:
                    similar_pairs[val2].append((val1, similar))
                else:
                    similar_pairs[val2] = [(val1, similar)]
    def Pearson_Prediction(id):
        # Handling Cold Start Problem #
        if id[0] not in user_dict:
            return (id, business_average[business_dict[id[1]]])
        elif id[1] not in business_dict:
            return (id, user_average[user_dict[id[0]]])
        elif id[0] not in user_dict and id[1] not in business_dict:
            return (id, 2.75)
        else:
            closest_pairs = similar_pairs[user_dict[id[0]]]
            filter_users = 0
            closest_pairs.sort(key=lambda a: a[1], reverse=True)
            closest = list()
            near_par = 5
            if len(closest_pairs) > near_par:
                closest = closest_pairs[:5]   # Taking the nearest neighbours #
            else:
                closest = closest_pairs
            mean = user_average[user_dict[id[0]]]
            nume = 0
            denom = 0
            if len(closest) == 0:
                return (id, mean)
            for i in closest:
                if i[0] in ui_dict[business_dict[id[1]]]:
                    rate_val = ui_dict[business_dict[id[1]]][i[0]] - user_average[i[0]]
                    nume = nume + rate_val * i[1]
                    denom = denom + abs(i[1])
            if denom == 0:
                return (id, mean)
            final = nume/denom + mean
            return (id, final)
    final_pred = new_validation_data.map(Pearson_Prediction).collect()
    opened_file = open(sys.argv[4], 'a+')
    opened_file.write("user_id, business_id, prediction\n")
    for id in final_pred:
            val = id[0][0] + "," + id[0][1] + "," + str(id[1]) + "\n"
            opened_file.write(val)
    opened_file.close()

# Item Based Collaborative Filtering #
if case == 3:
        business_to_user = ui_data.map(lambda a: (a[0][1], {a[0][0]})).reduceByKey(lambda a, b: a.union(b)).collect()
        bu_dict = dict()
        for val in business_to_user:
            bu_dict[val[0]] = val[1]
        # Pearson's correlation for Item based CF #
        def Pearson_Correlation(val1, val2):
            co_rated = bu_dict[val1].intersection(bu_dict[val2])
            co_rated_length = len(co_rated)
            if co_rated_length == 0:
                return 0
            business1_weight = []
            business2_weight = []
            for x in co_rated:
                business1_weight.append(ui_dict[val1][x])
                business2_weight.append(ui_dict[val2][x])
            business1 = len(business1_weight)
            business2 = len(business2_weight)
            business1_avg = sum(business1_weight) / business1
            business2_avg = sum(business2_weight) / business2
            nume = 0
            denom1 = 0
            denom2 = 0
            business1_count = len(business1_weight)
            for x in range(0,business1_count):
                nume1 = business1_weight[x] - business1_avg
                nume2 = business2_weight[x] - business2_avg
                nume = nume + nume1 * nume2
                denom1 = denom1 + nume1 ** 2
                denom2 = denom2 + nume2 ** 2
            denom1 =  math.sqrt(denom1)
            denom2 =  math.sqrt(denom2)
            denom = denom1 * denom2
            if nume == 0:
                return 0
            final_weight = nume/denom
            return final_weight

        similar_pairs = {}
        businessid_count = len(inverse_business_mapping)
        for val1 in range(0,businessid_count):
            similar_pairs[val1] = list()
            for val2 in range(val1 + 1, businessid_count):
                similar = Pearson_Correlation(val1, val2)
                if similar != 0:
                    sim_count = 0
                    similar_pairs[val1].append((val2, similar))

        def Pearson_Prediction(id):
            # Handling Cold Start Problem #
            if id[0] not in user_dict:
                return (id, business_average[business_dict[id[1]]])
            elif id[1] not in business_dict:
                return (id, user_average[user_dict[id[0]]])
            elif id[0] not in user_dict and id[1] not in business_dict:
                return (id, 2.75)
            else:
                closest_pairs = similar_pairs[business_dict[id[1]]]
                filter_business = 0
                closest_pairs.sort(key=lambda a: a[1], reverse=True)
                closest = list()
                near_par = 5
                if len(closest_pairs) > near_par:
                    closest = closest_pairs[:5]   # Taking the nearest neighbours #
                else:
                    closest = closest_pairs
                mean = business_average[business_dict[id[1]]]
                nume = 0
                denom = 0
                if len(closest) == 0:
                    return (id, mean)
                for i in closest:
                    if user_dict[id[0]] in ui_dict[i[0]]:
                        rate_val = ui_dict[i[0]][user_dict[id[0]]] - business_average[i[0]]
                        nume = nume + rate_val * i[1]
                        denom = denom + abs(i[1])
                if denom == 0:
                    return (id, mean)
                final = nume / denom + mean
                return (id, final)
        final_pred = new_validation_data.map(Pearson_Prediction).collect()
        opened_file = open(sys.argv[4], "a+")
        opened_file.write("user_id, business_id, prediction\n")
        for id in final_pred:
                val = id[0][0] + "," + id[0][1] + "," + str(id[1]) + "\n"
                opened_file.write(val)
        opened_file.close()
end = time.time()
#print("Duration: ", end - start)

# RMSE Calculation #
#ground_truth = sc.textFile(validation_file_path).map(lambda a: a.split(',')).filter(lambda a: a[0] != 'user_id').map(lambda a: ((a[0], a[1]), float(a[2])))

#predicted_val = sc.textFile(sys.argv[4]).map(lambda a: a.split(',')).filter(lambda a: a[0] != 'user_id').map(lambda a: ((a[0], a[1]), float(a[2])))

#ratesAndPreds = ground_truth.join(predicted_val)

#MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2)
#print("Mean Squared Error = " + str(MSE.mean() ** 0.5))










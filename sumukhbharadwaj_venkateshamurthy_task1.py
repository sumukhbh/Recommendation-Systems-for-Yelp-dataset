from pyspark import SparkContext
import sys
import time
from itertools import combinations
sc = SparkContext("local[*]", "Sumukh_Task1")
sc.setLogLevel("OFF")
SparkContext.setSystemProperty('spark.executor.memory', '4g')
SparkContext.setSystemProperty('spark.driver.memory', '4g')
input_file_path = sys.argv[1]
start = time.time()
RDD_inter = sc.textFile(input_file_path)
header = RDD_inter.first()
result = RDD_inter.filter(lambda a: a != header)
result_RDD = result.map(lambda a: a.split(","))
user_dict = result_RDD.map(lambda a: a[0]).zipWithIndex().collectAsMap()
#print(user_dict)
no_of_users = result_RDD.map(lambda a: a[0]).distinct()
user_count = no_of_users.count()
#print(no_of_users.take(5))
user_business_RDD = result_RDD.map(lambda a: (a[1], user_dict.get(a[0]))).groupByKey().sortByKey().mapValues(lambda a: list(a))
user_business = user_business_RDD.collect()
#print(user_business.take(5))
characteristic_matrix = dict()
for business in user_business:
     characteristic_matrix[business[0]] = business[1] # mapping the users who reviewed each business_id #
#print(characteristic_matrix)
support = 0.5
no_of_bands = 40
no_of_rows = 2
no_of_hashes = 80  # bands multiplied by rows #

def create_minhash(user_data):
    user_list = list(user_data)
    initial_value = float('Inf')
    value_in_cols = [initial_value for x in range(no_of_hashes)]
    for user in user_data:
        for x in range(1, no_of_hashes + 1):
            hash_value = ((x * user) + (x * 53)) % user_count
            value_in_cols[x - 1] = min(hash_value, value_in_cols[x - 1])
    return value_in_cols

candidates_signature = user_business_RDD.mapValues(create_minhash)
#print(candidates_signature.take(5))

def LSH(id, signatures):
    lsh_pairs = []
    for band in range(no_of_bands):
        band_name = band
        lsh_signature = signatures[band*no_of_rows:(band*no_of_rows)+no_of_rows]
        lsh_signature.insert(0, band_name)
        pairs = (tuple(lsh_signature), id)
        lsh_pairs.append(pairs)
        #print(lsh_pairs)
    return lsh_pairs

candidates_LSH = candidates_signature.flatMap(lambda x: LSH(x[0], x[1]))
#print(candidates_LSH.take(5))

def similar_businesses(business):
  similar_pairs = list()
  similar_pairs = sorted(list(combinations(sorted(business),2)))
  no_of_business = len(business)
  #print(no_of_business)
  return similar_pairs

similar_candidates = candidates_LSH.groupByKey().filter(lambda x: len(list(x[1])) > 1).flatMap(lambda x: similar_businesses(sorted(list(x[1])))).distinct()

def calculate_jaccard(candidate_pairs, charactertistic_matrix):

    set_1 = set(characteristic_matrix.get(candidate_pairs[0]))
    set_2 = set(charactertistic_matrix.get(candidate_pairs[1]))
    union_calculation = len(set_1.union(set_2))
    intersection_calculation = len(set_1.intersection(set_2))
    jaccard_similarity = float(intersection_calculation)/(union_calculation)
    return (candidate_pairs, jaccard_similarity)

similar_final = similar_candidates.map(lambda x: calculate_jaccard(x, characteristic_matrix)).filter(lambda x: x[1] >= support).sortByKey()
result = similar_final.collect()
#print("Count: ", len(result))
opened_file = open(sys.argv[2], "a+")
opened_file.write("business_id_1, business_id_2, similarity\n")
for pair in result:
    val = pair[0][0]+","+ pair[0][1]+ "," +str(pair[1]) + "\n"
    opened_file.write(val)
opened_file.close()

#print(result)
end = time.time()
#print("Duration: ", end-start)

# ground = sc.textFile("pure_jaccard_similarity.csv").map(lambda x: x.split(",")).map(lambda x: (x[0], x[1]))
#
# rrr = similar_final.map(lambda x: (x[0][0], x[0][1]))
# rrrrr = list(rrr.collect())
# ggggg = list(ground.collect())
# tp = rrr.intersection(ground)
# ttttt= list(tp.collect())
#
# precision = len(ttttt)/len(rrrrr)
# recall = len(ttttt)/len(ggggg)
# print("precision:")
# print(precision)
# print("recall:")
# print(recall)




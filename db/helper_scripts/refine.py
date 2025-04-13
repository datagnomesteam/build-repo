import json
import csv

with open('openrefine.json') as f:
    ops = json.load(f)

with open('name_mapping.csv', 'w', newline='') as out:
    writer = csv.writer(out)
    writer.writerow(['original_name', 'normalized_name'])
    for op in ops:
        for cluster in op['edits']:
            to = cluster['to']
            for original in cluster['from']:
                writer.writerow([original, to])

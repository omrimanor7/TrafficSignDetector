

def main():
    f_classes = open("FullIJCNN2013/classes.txt", "r")
    classes = ""
    class_id_to_class_name = {}
    for line in f_classes:
        line_arr = line.split(' = ')
        line_arr[1] = line_arr[1].replace(' ', '_').rstrip()
        class_id = line_arr[0]
        class_name = line_arr[1]
        class_id_to_class_name[class_id] = class_name
        classes += class_name + ',' + class_id + '\n'
    classes = classes.rstrip()
    f_classes.close()

    f_annotations = open("FullIJCNN2013/annotations.txt", "r")
    annotations = ""
    for line in f_annotations:
        line_arr = line.split(';')
        line_arr[5] = class_id_to_class_name[line_arr[5].rstrip()] + '\n'
        annotations += ",".join(line_arr)
    f_annotations.close()

    # write files to csv
    with open('FullIJCNN2013/classes.csv', 'w') as classes_csv:
        classes_csv.write(classes)
    with open('FullIJCNN2013/annotations.csv', 'w') as annotations_csv:
        annotations_csv.write(annotations)


if __name__ == '__main__':
    main()

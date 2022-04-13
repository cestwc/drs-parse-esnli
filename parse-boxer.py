import os
import argparse

# os.environ["CUDADEVICE_ORDER"] = "PCIBUS_ID"

parser = argparse.ArgumentParser(description="esnli")
parser.add_argument("--split", type=str, default='train', help='which split of dataset')
parser.add_argument("--shard", type=int, default=128, help="divide the dataset into")
parser.add_argument("--index", type=int, default=0, help="which partition")
parser.add_argument("--dataPath", type=str, default='esnli', help='path of files to process')
parser.add_argument("--save", type=str, default='esnli_boxer', help='path to save processed data')
parser.add_argument('--shard', action="store_true", help='use DnCNN as reference?')
# parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
# parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")

opt = parser.parse_args()

def drs_parse(e):
	try:
		e['hypothesis_logic'] = my_boxer.interpret(' '.join(word_tokenize(sample['hypothesis']))).fol().__str__()
	except:
		e['hypothesis_logic'] = '(x)'
		print('hypothesis')
	try:
		e['premise_logic'] = my_boxer.interpret(' '.join(word_tokenize(sample['premise']))).fol().__str__()
	except:
		e['premise_logic'] = '(x)'
		print('premise')
	return e

def main():
	esnli = load_from_disk(opt.dataPath)

	snli_parsed = snli.map(drs_parse)

	raw_data = load_dataset('esnli')[opt.split].shard(opt.shard, opt.index)	
	snli_parsed = raw_data.map(drs_parse)
	snli_parsed.save_to_disk(f"{opt.save}/{opt.split}_{opt.shard}_{opt.index}")


if __name__ == "__main__":
	main()

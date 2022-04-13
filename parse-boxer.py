import os
import argparse

# os.environ["CUDADEVICE_ORDER"] = "PCIBUS_ID"

parser = argparse.ArgumentParser(description="esnli")
parser.add_argument("--split", type=str, default='train', help='which split of dataset')
parser.add_argument("--shard", type=int, default=128, help="divide the dataset into")
parser.add_argument("--index", type=int, default=0, help="which partition")
parser.add_argument("--dataPath", type=str, default='esnli', help='path of files to process')
parser.add_argument("--save", type=str, default='esnli_boxer', help='path to save processed data')
# parser.add_argument('--shard', action="store_true", help='use DnCNN as reference?')
# parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
# parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")

opt = parser.parse_args()

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

import os
import tempfile
import nltk.sem.boxer

class Boxer_(nltk.sem.boxer.Boxer):
	def __init__(
		self,
		boxer_drs_interpreter=None,
		elimeq=False,
		bin_dir='candc/candc/bin',
		verbose=False,
		resolve=True,
	):
		super(Boxer_, self).__init__(boxer_drs_interpreter, elimeq, bin_dir, verbose, resolve)

	def _call_boxer(self, candc_out, verbose=False):
		"""
		Call the ``boxer`` binary with the given input.

		:param candc_out: str output from C&C parser
		:return: stdout
		"""
		f = None
		try:
			fd, temp_filename = tempfile.mkstemp(
				prefix="boxer-", suffix=".in", text=True
			)
			f = os.fdopen(fd, "w")
			f.write(candc_out.decode("utf-8") )
		finally:
			if f:
				f.close()

		args = [
			"--box",
			"false",
			"--semantics",
			"drs",
			#'--flat', 'false', # removed from boxer
			"--resolve",
			["false", "true"][self._resolve],
			"--elimeq",
			["false", "true"][self._elimeq],
			"--format",
			"prolog",
			"--instantiate",
			"true",
			"--input",
			temp_filename,
		]
		stdout = self._call(None, self._boxer_bin, args, verbose)
		os.remove(temp_filename)
		return stdout

	def _parse_to_drs_dict(self, boxer_out, use_disc_id):
		lines = boxer_out.decode("utf-8").split("\n")
		drs_dict = {}
		i = 0
		while i < len(lines):
			line = lines[i]
			if line.startswith("id("):
				comma_idx = line.index(",")
				discourse_id = line[3:comma_idx]
				if discourse_id[0] == "'" and discourse_id[-1] == "'":
					discourse_id = discourse_id[1:-1]
				drs_id = line[comma_idx + 1 : line.index(")")]
				i += 1
				line = lines[i]
				assert line.startswith(f"sem({drs_id},")
				if line[-4:] == "').'":
					line = line[:-4] + ")."
				assert line.endswith(")."), f"can't parse line: {line}"

				search_start = len(f"sem({drs_id},[")
				brace_count = 1
				drs_start = -1
				for j, c in enumerate(line[search_start:]):
					if c == "[":
						brace_count += 1
					if c == "]":
						brace_count -= 1
						if brace_count == 0:
							drs_start = search_start + j + 1
							if line[drs_start : drs_start + 3] == "','":
								drs_start = drs_start + 3
							else:
								drs_start = drs_start + 1
							break
				assert drs_start > -1

				drs_input = line[drs_start:-2].strip()
				parsed = self._parse_drs(drs_input, discourse_id, use_disc_id)
				drs_dict[discourse_id] = self._boxer_drs_interpreter.interpret(parsed)
			i += 1
		return drs_dict
	
my_boxer = Boxer_()

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

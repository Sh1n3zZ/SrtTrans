import re
from dataclasses import dataclass
from typing import List
from datetime import datetime
import os
import argparse
import asyncio
from tqdm import tqdm
from endpoint.endpoint import OpenAITranslator

@dataclass
class SubtitleEntry:
    index: int
    start_time: str
    end_time: str
    content: str
    translated_content: str = ""

class SrtParser:
    def __init__(self):
        self.entries: List[SubtitleEntry] = []
        self.source_language: str = ""  # source language
        self.translations: dict[str, List[str]] = {}  # translation results by target language
    
    def parse_time(self, time_str: str) -> tuple[str, datetime]:
        """parse timestamp string, return original string and datetime object tuple"""
        time_str = time_str.strip()
        # convert timestamp to datetime object for comparison
        time_parts = time_str.replace(',', '.').split(':')
        hours = int(time_parts[0])
        minutes = int(time_parts[1])
        seconds = float(time_parts[2])
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return time_str, datetime.fromtimestamp(total_seconds)
    
    def parse_file(self, file_path: str, source_language: str = "") -> List[SubtitleEntry]:
        """parse srt file
        
        Args:
            file_path: srt file path
            source_language: source language (optional)
        """
        self.source_language = source_language
        self.entries.clear()
        self.translations.clear()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # split subtitle blocks by empty lines
        subtitle_blocks = content.strip().split('\n\n')
        
        # use tqdm to display parsing progress
        for block in tqdm(subtitle_blocks, desc="Parsing subtitles"):
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue
                
            # parse index
            index = int(lines[0])
            
            # parse timestamp
            time_line = lines[1]
            start_time_str, end_time_str = time_line.split(' --> ')
            start_time_str, start_time_obj = self.parse_time(start_time_str)
            end_time_str, _ = self.parse_time(end_time_str)
            
            # get subtitle text (may span multiple lines)
            text = '\n'.join(lines[2:])
            
            # create subtitle entry
            entry = SubtitleEntry(
                index=index,
                start_time=start_time_str,
                end_time=end_time_str,
                content=text
            )
            self.entries.append((start_time_obj, entry))
        
        # sort by start time
        self.entries.sort(key=lambda x: x[0])
        # update index and extract sorted entries
        sorted_entries = []
        for i, (_, entry) in enumerate(self.entries, 1):
            entry.index = i
            sorted_entries.append(entry)
        self.entries = sorted_entries
        
        return self.entries

    async def translate_entries(self, translator: OpenAITranslator, target_language: str = "中文", 
                              max_concurrency: int = 5, use_history: bool = False,
                              history_file: str = None) -> None:
        """async translate all subtitle entries
        
        Args:
            translator: translator instance
            target_language: target language
            max_concurrency: maximum concurrency
            use_history: whether to use history translation
            history_file: history translation file path
        """
        # if use history translation, load history file first
        history_map = {}  # original to translation mapping
        if use_history and history_file:
            try:
                history_parser = SrtParser()
                history_parser.parse_file(history_file)
                # create original to translation mapping
                for entry in history_parser.entries:
                    if entry.translated_content:  # only save entries with translation
                        # use original as key, translation as value
                        # remove possible whitespace and newline characters to improve matching rate
                        key = entry.content.strip()
                        history_map[key] = entry.translated_content.strip()
                print(f"Loaded {len(history_map)} history translation records")
            except Exception as e:
                print(f"Failed to load history translation file: {str(e)}")
                history_map = {}

        # prepare texts to translate
        texts_to_translate = []
        for entry in self.entries:
            # clean original text
            clean_content = entry.content.strip()
            # check if there is a matching history translation
            if history_map and clean_content in history_map:
                # use history translation
                entry.translated_content = history_map[clean_content]
            else:
                # add to translation list
                texts_to_translate.append(entry.content)

        # if there are still texts to translate, translate them
        if texts_to_translate:
            print(f"Need to translate {len(texts_to_translate)} new texts")
            if history_map:
                print(f"Reused {len(history_map) - len(texts_to_translate)} history translations")
            translated_texts = await translator.translate_batch(texts_to_translate, target_language, max_concurrency)
            
            # update translation results
            j = 0
            for entry in self.entries:
                if not entry.translated_content:  # if not using history translation
                    entry.translated_content = translated_texts[j]
                    j += 1
        else:
            print("All texts have been found in history translations")
        
        # store translation results
        self.translations[target_language] = [entry.translated_content for entry in self.entries]

    def save_translated_srt(self, output_path: str, separate_languages: bool = False,
                          target_language: str = None) -> None:
        """save subtitle file
        
        Args:
            output_path: output file path
            separate_languages: whether to save original and translation as separate files
            target_language: target language, used for file naming
        """
        if not separate_languages:
            # original merge save logic
            with open(output_path, 'w', encoding='utf-8') as f:
                for entry in tqdm(self.entries, desc="Saving subtitles"):
                    f.write(f"{entry.index}\n")
                    f.write(f"{entry.start_time} --> {entry.end_time}\n")
                    f.write(f"{entry.content}\n")
                    if entry.translated_content:
                        f.write(f"{entry.translated_content}\n")
                    f.write("\n")
            print(f"Saved merged subtitle file: {os.path.abspath(output_path)}")
        else:
            # separate save original and translation
            lang_suffix = f"_{self.source_language}" if self.source_language else ""
            original_path = output_path.rsplit('.', 1)[0] + f'{lang_suffix}_original.srt'
            translated_path = output_path.rsplit('.', 1)[0] + f'_{target_language}.srt'
            
            # save original
            with open(original_path, 'w', encoding='utf-8') as f:
                for entry in tqdm(self.entries, desc="Saving original subtitles"):
                    f.write(f"{entry.index}\n")
                    f.write(f"{entry.start_time} --> {entry.end_time}\n")
                    f.write(f"{entry.content}\n\n")
            print(f"Saved original subtitle file: {os.path.abspath(original_path)}")
            
            # if there is translation, save translation
            if any(entry.translated_content for entry in self.entries):
                with open(translated_path, 'w', encoding='utf-8') as f:
                    for entry in tqdm(self.entries, desc="Saving translated subtitles"):
                        if entry.translated_content:  # skip untranslated entries
                            f.write(f"{entry.index}\n")
                            f.write(f"{entry.start_time} --> {entry.end_time}\n")
                            f.write(f"{entry.translated_content}\n\n")
                print(f"Saved translated subtitle file: {os.path.abspath(translated_path)}")
            else:
                print("No translation found, skipping save translation file")

    def split_translated_file(self, file_path: str) -> None:
        """split translated srt file to original and translation
        
        Args:
            file_path: input translated srt file path
        """
        self.entries.clear()  # clear existing entries
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # split subtitle blocks by empty lines
        subtitle_blocks = content.strip().split('\n\n')
        
        # use tqdm to display parsing progress
        for block in tqdm(subtitle_blocks, desc="Parsing subtitles"):
            lines = block.strip().split('\n')
            if len(lines) < 4:  # at least 4 lines (index, time, original, translation)
                continue
            
            # parse index
            index = int(lines[0])
            
            # parse timestamp
            time_line = lines[1]
            start_time, end_time = time_line.split(' --> ')
            
            # get original and translation
            content = lines[2]
            translated_content = lines[3] if len(lines) > 3 else ""
            
            # create subtitle entry
            entry = SubtitleEntry(
                index=index,
                start_time=start_time.strip(),
                end_time=end_time.strip(),
                content=content,
                translated_content=translated_content
            )
            self.entries.append(entry)
        
        # save separated file
        output_base = file_path.rsplit('.', 1)[0]
        self.save_translated_srt(output_base + '.srt', separate_languages=True)

async def async_main(args: argparse.Namespace) -> None:
    # initialize parser
    parser = SrtParser()

    try:
        if args.split_file:
            # split translated file mode
            print(f"\nStart splitting file: {args.input_file}")
            parser.split_translated_file(args.input_file)
            print("\nSplit completed!")
            return

        # parse srt file
        print(f"\nStart processing file: {args.input_file}")
        subtitles = parser.parse_file(args.input_file, args.source_language)
        print(f"Found {len(subtitles)} subtitles")
        
        if not args.sort_only:
            # initialize translator
            try:
                translator = OpenAITranslator(api_key=args.api_key, api_base=args.api_base)
            except ValueError as e:
                print(f"Error: {str(e)}")
                print("Please set OPENAI_API_KEY environment variable or use --api-key parameter to provide API key")
                return
            
            # translate subtitles
            print(f"\nStart translating (target language: {args.target_language}, concurrency: {args.max_concurrency})")
            await parser.translate_entries(
                translator, 
                args.target_language, 
                args.max_concurrency,
                args.use_history,
                args.history_file
            )
            output_file = args.input_file.rsplit('.', 1)[0] + f'_translated_{args.target_language}.srt'
        else:
            # sort only mode
            output_file = args.input_file.rsplit('.', 1)[0] + '_sorted.srt'
        
        # save file
        print(f"\nSaving results...")
        parser.save_translated_srt(output_file, args.separate_output, args.target_language)
        print("\nProcessing completed!")
        
    except FileNotFoundError:
        print(f"Error: file not found '{args.input_file}'")
    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='SRT subtitle translation tool')
    parser.add_argument('input_file', help='input srt file path')
    parser.add_argument('--source-language', '-sl', default="",
                      help='source language (optional, for file naming)')
    parser.add_argument('--target-language', '-t', default='中文',
                      help='target language (default: Chinese)')
    parser.add_argument('--api-key', '-k', help='OpenAI API key')
    parser.add_argument('--api-base', '-b', default='https://api.chatnio.net/v1',
                      help='API base URL (default: https://api.chatnio.net/v1)')
    parser.add_argument('--max-concurrency', '-m', type=int, default=5,
                      help='maximum concurrency (default: 5)')
    parser.add_argument('--sort-only', '-s', action='store_true',
                      help='sort subtitles only, no translation')
    parser.add_argument('--separate-output', '-p', action='store_true',
                      help='save original and translation as separate srt files')
    parser.add_argument('--split-file', '-f', action='store_true',
                      help='split translated srt file to original and translation')
    parser.add_argument('--use-history', '-u', action='store_true',
                      help='use history translation')
    parser.add_argument('--history-file', '-hf',
                      help='history translation file path (used with --use-history)')
    args = parser.parse_args()
    
    # run async main program
    asyncio.run(async_main(args))

if __name__ == "__main__":
    main()

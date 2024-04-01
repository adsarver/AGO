import pandas as pd
import os
import sys

os.chdir(sys.argv[1])

template = 'https://images.hgmsites.net/'


def run():
    df = pd.read_csv('specs-and-pics.csv', dtype=str, index_col=0).T
    
    df = df.iloc[:, :-121]
    
    df = (df.melt(id_vars=df.columns[:3],value_name='Picture', value_vars=df.columns[3:])
          .set_index(['Picture']).reset_index())
    
    impt_cols = [
        'Make', 'Model', 'Year', 'Picture']
    
    df = df[impt_cols]
    
    df['Picture'] = template + df['Picture']

    df = df.dropna(subset=['Picture'])

    # ----- // Cleaning Columns

    df['ID'] = df.iloc[:, :-1].apply(lambda x: '_'.join(x.astype(str)), axis=1)

    final_df = df[['ID', 'Picture']]

    final_df.to_csv('id_and_pic_url.csv', index=None)


if __name__ == '__main__':
    print('%s started running.' % os.path.basename(__file__))
    run()
    print('%s finished running.' % os.path.basename(__file__))

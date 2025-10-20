import pandas as pd
import numpy as np
import os

def create_and_save_all_data(num_days=180, num_couriers=15, num_zones=5):
    """
    프로젝트에 필요한 4가지 가상 데이터 파일을 생성하고 저장합니다.
    """
    np.random.seed(42)
    os.makedirs('data/raw', exist_ok=True)
    
    #기사 프로필
    courier_ids = [f'C_{i:02d}' for i in range(1, num_couriers + 1)]
    couriers_df = pd.DataFrame({
        'courier_id': courier_ids,
        'skill': np.random.choice(['초보자', '경력자', '숙련자'], num_couriers, p=[0.3, 0.4, 0.3]),
        'home_lat': np.random.uniform(37.4, 37.6, num_couriers).round(5),
        'home_lng': np.random.uniform(126.9, 127.1, num_couriers).round(5)
    })
    couriers_df.to_csv('data/raw/couriers.csv', index=False)
    print("✅ data/raw/couriers.csv 생성 완료")

    #지역 정보 및 수요
    zones_df = pd.DataFrame({
        'zone_id': [f'Z_{i}' for i in range(1, num_zones + 1)],
        'zone_lat': np.random.uniform(37.4, 37.6, num_zones).round(5),
        'zone_lng': np.random.uniform(126.9, 127.1, num_zones).round(5),
        'demand_qty': np.random.randint(200, 300, num_zones) # 구역별 총 수요 (내일 수요)
    })
    zones_df.to_csv('data/raw/zones.csv', index=False)
    print("✅ data/raw/zones.csv 생성 완료")

    #일일 데이터
    all_data = []
    
    for courier_id in courier_ids:
        df = pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=num_days, freq='D'),
            'courier_id': courier_id,
            
            'work_hours': np.random.uniform(7, 11, num_days).round(1),
            'deliveries': np.random.randint(80, 130, num_days), # 어제 처리 물량
            'bmi': np.random.uniform(20, 28, num_days).round(1),
            'avg_hr': np.random.uniform(70, 100, num_days).round(1),
            'steps': np.random.randint(15000, 30000, num_days),
        })
        
        df['load_rel'] = np.random.choice([-1, 0, 1], size=num_days, p=[0.2, 0.6, 0.2])
        df['strain'] = np.random.choice([0.0, 0.5, 1.0], size=num_days, p=[0.5, 0.3, 0.2])
        df['wish'] = np.random.choice([-1, 0, 1], size=num_days, p=[0.2, 0.6, 0.2])
        
        all_data.append(df)
        
    full_df = pd.concat(all_data, ignore_index=True)
    
    # 날짜 형식 조정
    full_df['date'] = full_df['date'].dt.strftime('%Y-%m-%d')

    #업무/신체
    metrics_cols = ['date', 'courier_id', 'work_hours', 'deliveries', 'bmi', 'avg_hr', 'steps']
    full_df[metrics_cols].to_csv('data/raw/daily_metrics.csv', index=False)
    print("✅ data/raw/daily_metrics.csv 생성 완료")

    #설문
    survey_cols = ['date', 'courier_id', 'load_rel', 'strain', 'wish']
    full_df[survey_cols].to_csv('data/raw/daily_surveys.csv', index=False)
    print("✅ data/raw/daily_surveys.csv 생성 완료")


if __name__ == '__main__':
    print("--- 🛠️ AI 시스템용 4가지 가상 데이터 생성 시작 ---")
    create_and_save_all_data()
    print("--------------------------------------------------")
    print("🎉 모든 데이터 준비 완료. 이제 model_trainer.py를 실행할 수 있습니다.")
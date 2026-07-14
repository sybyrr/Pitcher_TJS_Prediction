import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PAINS | MLB TJS 위험 순위 연구 대시보드",
  description: "동결된 MLB 투수 TJS 위험 모델의 과거 재현 결과와 기여 요인을 검토하는 연구용 대시보드",
  openGraph: {
    title: "PAINS | MLB TJS 위험 순위 연구 대시보드",
    description: "동결된 과거 재현 결과를 검토하는 비임상 연구 대시보드",
    type: "website",
    locale: "ko_KR",
    images: [{
      url: "/og-research.png",
      width: 1731,
      height: 909,
      alt: "추상적인 투구 위치와 시간 신호를 표현한 PAINS 연구 그래픽",
    }],
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}

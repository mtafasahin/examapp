import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable, of, switchMap, tap } from 'rxjs';
import { CheckStudentResponse } from '../models/check-student-response';
import { Router } from '@angular/router';
import { Subject } from '../models/subject';
import { Topic } from '../models/topic';
import { SubTopic } from '../models/subtopic';
import { InstanceSummary, Exam, Paged, Test, TestInstance } from '../models/test-instance';
import { StudentAnswer } from '../models/student-answer';
import { AnswerChoice, QuestionRegion } from '../models/draws';
import { Answer } from '../models/answer';
import { StudentStatisticsResponse } from '../models/statistics';

@Injectable({
  providedIn: 'root'
})
export class TestService {

  private baseUrl = '/api/worksheet';

  constructor(private http: HttpClient, private router: Router) {}

  getTestWithAnswers(testInstanceId: number): Observable<TestInstance> {
    return this.http.get<TestInstance>(`/api/worksheet/test-instance/${testInstanceId}`);
  }

  getCanvasTestWithAnswers(testInstanceId: number): Observable<TestInstance> {
    return this.http.get<TestInstance>(`/api/worksheet/test-canvas-instance/${testInstanceId}`);
  }

  getCanvasTestResults(testInstanceId: number): Observable<TestInstance> {
    return this.http.get<TestInstance>(`/api/worksheet/test-canvas-instance-result/${testInstanceId}`);
  }

  saveAnswer( studentAnswer: StudentAnswer ): Observable<any> {
    return this.http.post<any>(`/api/worksheet/save-answer`,  studentAnswer );
  }

  completeTest( testInstanceId: number ): Observable<any> {
    return this.http.put<any>(`/api/worksheet/end-test/${testInstanceId}`, null);
  }

  startTest( testInstanceId: number ): Observable<any> {
    return this.http.post<any>(`/api/worksheet/start-test/${testInstanceId}`, null);
  }

  loadTest() {
    return this.http.get<Exam[]>(`/api/worksheet`);
  }

  search(query: string | undefined, subjectIds: number[], gradeId?: number, pageNumber: number = 1, pageSize= 10, booktestId=0): 
    Observable<Paged<Test>> {
    const subjectIdsParam = subjectIds && subjectIds.length > 0 ? '&subjectIds=' + subjectIds.join(`&subjectIds=`) : '';
    const gradeIdParam = gradeId ? `&gradeId=${gradeId}` : '';
    const bookTestIdParam = booktestId ? `&bookTestId=${booktestId}` : '';
    var reqUrl = `/api/worksheet/list?search=${query}&pageNumber=${pageNumber}&pageSize=${pageSize}${subjectIdsParam}${gradeIdParam}${bookTestIdParam}`;
    console.log(reqUrl);
    return this.http.get<Paged<Test>>(reqUrl);
  }

  getLatest(pageNumber: number = 1, pageSize= 10): Observable<Test[]> {
    return this.http.get<Test[]>(`/api/worksheet/latest?pageNumber=${pageNumber}&pageSize=${pageSize}`);
  }

  getCompleted(pageNumber: number = 1): Observable<Paged<InstanceSummary>> {
    return this.http.get<Paged<InstanceSummary>>(`/api/worksheet/CompletedTests?pageNumber=${pageNumber}&pageSize=10`);
  }
  
  create(test: Test): Observable<{message:string, examId: number}> {
    return this.http.post<any>('/api/worksheet', test);
  }

  get(id: number): Observable<Test> {
      var reqUrl = `/api/worksheet/list?id=${id}`;
      return this.http.get<Paged<Test>>(reqUrl).pipe(
        switchMap(response => {
          return of(response.items[0]);
        }
      ));
  }

  convertAnswerToAnswerChoice(answer: Answer) : AnswerChoice {
    return {
          id: answer.id,
          label: answer.text,
          x: answer.x,
          y: answer.y,
          width: answer.width,
          height: answer.height
    }
  }

  convertTestInstanceToRegions(testInstance: TestInstance): QuestionRegion[] 
  {
      if (!testInstance || !testInstance.testInstanceQuestions || testInstance.testInstanceQuestions.length === 0) {
          return [];
      }

      const imageSrc = testInstance.testInstanceQuestions[0].question.imageUrl;

      const regions: QuestionRegion[] = testInstance.testInstanceQuestions.map(qInstance => {
          const question = qInstance.question;
          return {
              id: question.id,
              name: `Soru ${qInstance.order}`,
              x: question.x,
              y: question.y,
              width: question.width,
              height: question.height,
              isExample: question.isExample,
              passageId: question.passage ? question.passage.id.toString() : '',
              imageId: question.imageUrl,
              imageUrl: question.imageUrl,
              exampleAnswer: question.isExample ? question.practiceCorrectAnswer : null,
              answers: question.answers.map(answer => ({
                  id: answer.id,
                  label: answer.text,
                  x: answer.x,
                  y: answer.y,
                  width: answer.width,
                  height: answer.height
              })),
              passage: question.passage ? {
                  id: question.passage.id,
                  title: question.passage.title,
                  x: question.passage.x,
                  y: question.passage.y,
                  width: question.passage.width,
                  height: question.passage.height
              } : undefined
          };
      });

      return regions;
  }

  studentStatistics(): Observable<StudentStatisticsResponse> {
    return this.http.get<StudentStatisticsResponse>(`${this.baseUrl}/student/statistics`);      
  }

}
import { Routes } from '@angular/router';
import { RegisterComponent } from './pages/register/register.component';
import { LoginComponent } from './pages/login/login.component';
import { StudentRegisterComponent } from './pages/student-register/student-register.component';
import { authGuard } from './shared/guards/auth.guard';
import { QuestionComponent } from './pages/question/question.component';
import { QuestionViewComponent } from './pages/question-view/question-view.component';
import { StudentProfileComponent } from './pages/student-profile/student-profile.component';
import { TestCreateComponent } from './pages/test-create/test-create.component';
import { HomeComponent } from './home/home.component';
import { WorksheetListComponent } from './pages/worksheet-list/worksheet-list.component';
import { worksheetListResolver } from './shared/resolvers/worksheet-resolver';
import { ImageSelectorComponent } from './pages/image-selector/image-selector.component';
import { QuestionCanvasComponent } from './pages/question/question-canvas.component';
import { TestSolveCanvasComponent } from './pages/test-solve/test-solve-canvas.component';
import { WorksheetDetailComponent } from './pages/worksheet-detail/worksheet-detail.component';

export const routes: Routes = [
  { path: 'register', component: RegisterComponent },
  { path: 'login', component: LoginComponent },
  // { path: 'home', 
  //   component: WorksheetListComponent , 
  //   canActivate: [authGuard],
  //   resolve: { worksheets : worksheetListResolver }
  // },

  { path: 'student-register', component: StudentRegisterComponent, canActivate: [authGuard] },
  { path: 'question/:id', component: QuestionComponent, canActivate: [authGuard] },
  { path: 'questioncanvas', component: QuestionCanvasComponent, canActivate: [authGuard] },
  { path: 'questioncanvas/:id', component: QuestionCanvasComponent, canActivate: [authGuard] },
  { path: 'question', component: QuestionComponent, canActivate: [authGuard] },
  { path: 'imageselect', component: ImageSelectorComponent, canActivate: [authGuard] },
  { path: 'tests', component: WorksheetListComponent , 
    canActivate: [authGuard],
    resolve: { worksheets : worksheetListResolver } },
  { path: 'questions/view', component: QuestionViewComponent, canActivate: [authGuard] },
  { path: 'testsolve/:testInstanceId', component: TestSolveCanvasComponent, canActivate: [authGuard] },  // 🆕 Test çözme sayfası
  { path: 'test/:testId', component: WorksheetDetailComponent, canActivate: [authGuard] },  // 🆕 Test çözme sayfası
  { path: 'student-profile', component: StudentProfileComponent, canActivate: [authGuard] },  // 🆕 Test çözme sayfası
  { path: 'exam', component: TestCreateComponent, canActivate: [authGuard] },  // 🆕 Test çözme sayfası
  { path: 'exam/:id', component: TestCreateComponent, canActivate: [authGuard] },  // 🆕 Test çözme sayfası
  { path: '**', redirectTo: 'login' }
];

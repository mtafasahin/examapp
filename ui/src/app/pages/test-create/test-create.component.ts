import { CommonModule } from '@angular/common';
import { Component, ElementRef, inject, OnInit, ViewChild } from '@angular/core';
import { FormBuilder, FormGroup, FormsModule, ReactiveFormsModule, Validators } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatOptionModule } from '@angular/material/core';
import { MatFormFieldModule, MatLabel } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatToolbarModule } from '@angular/material/toolbar';
import { ActivatedRoute, NavigationExtras, Router } from '@angular/router';
import { Exam, Test } from '../../models/test-instance';
import { MatSelectModule } from '@angular/material/select';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { TestService } from '../../services/test.service';
import { WorksheetCardComponent } from '../worksheet-card/worksheet-card.component';
import { BookService } from '../../services/book.service';
import { Book, BookTest } from '../../models/book';
import { QuillEditorComponent } from 'ngx-quill';
import { SafeHtmlPipe } from '../../services/safehtml';
import { MatButtonModule } from '@angular/material/button';
import { AutofocusDirective } from '../../shared/directives/auto-focus.directive';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatTabsModule } from '@angular/material/tabs';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import * as XLSX from 'xlsx';
@Component({
  selector: 'app-test-create',
  templateUrl: './test-create.component.html',
  standalone: true,
  imports: [
    CommonModule,
    MatCardModule,
    MatButtonModule,
    FormsModule,
    MatFormFieldModule,
    MatOptionModule,
    MatInputModule,
    MatSelectModule,
    MatCheckboxModule,
    AutofocusDirective,
    MatIconModule,
    ReactiveFormsModule,
    MatToolbarModule,
    MatLabel,
    WorksheetCardComponent,
    MatProgressBarModule,
    MatTabsModule,
    MatSnackBarModule,
  ],
  styleUrls: ['./test-create.component.scss'],
})
export class TestCreateComponent implements OnInit {
  id!: number | null;
  exam!: Test;
  isEditMode: boolean = false;
  testForm!: FormGroup;
  gradeId: number = 1;
  books: Book[] = [];
  bookTests: BookTest[] = [];
  bookService = inject(BookService);
  grades = [
    { id: 1, name: '1. Sınıf' },
    { id: 2, name: '2. Sınıf' },
    { id: 3, name: '3. Sınıf' },
  ];
  showAddBookInput = false;
  showAddBookTestInput = false;
  /** grab the input once it’s in the DOM */
  @ViewChild('newBookInput') newBookInput!: ElementRef<HTMLInputElement>;
  selectedBulkFile: File | null = null;
  isUploading = false;
  bulkImportResults: any = null;

  constructor(
    private fb: FormBuilder,
    private router: Router,
    private route: ActivatedRoute,
    private testService: TestService
  ) {
    this.testForm = this.fb.group({
      name: ['', Validators.required],
      description: [''],
      gradeId: ['', Validators.required],
      maxDurationMinutes: [600, [Validators.required, Validators.min(10)]], // Varsayılan 10 dakika
      isPracticeTest: [false], // Çalışma testi mi?
      subtitle: [''],
      imageUrl: [''],
      newBookName: [''],
      newBookTestName: [''],
      bookTestId: [''],
      bookId: [''],
      questionCount: [0],
    });
  }

  ngOnInit(): void {
    this.id = this.route.snapshot.paramMap.get('id') ? Number(this.route.snapshot.paramMap.get('id')) : null;
    this.isEditMode = this.id !== null;
    this.reload();
  }

  reload() {
    this.showAddBookInput = false;
    this.showAddBookTestInput = false;
    this.loadBooks();
    if (this.isEditMode) {
      this.loadTest();
    }
  }

  loadTest() {
    this.testService.get(this.id!).subscribe((exam) => {
      this.onBookChange(exam.bookId);
      this.testForm.patchValue({
        name: exam.name,
        description: exam.description,
        gradeId: exam.gradeId,
        maxDurationMinutes: exam.maxDurationSeconds / 60,
        isPracticeTest: exam.isPracticeTest,
        subtitle: exam.subtitle,
        imageUrl: exam.imageUrl,
        bookTestId: exam.bookTestId,
        bookId: exam.bookId,
        questionCount: exam.questionCount,
      });
    });
  }

  loadBooks() {
    this.bookService.getAll().subscribe((data) => {
      this.books = data;
      if (this.testForm.value.bookId) {
        this.onBookChange(this.testForm.value.bookId);
      }
    });
  }

  onBookChange($event: any) {
    if ($event) {
      this.showAddBookInput = false;
      this.bookService.getTestsByBook($event).subscribe((data) => {
        this.bookTests = data;
      });
    }
  }

  openNewBookAdd() {
    this.showAddBookInput = true;
    this.testForm.patchValue({ bookId: null });
    this.bookTests = [];

    this.openNewBookTestAdd();
  }

  onNewBookBlur() {
    const name = this.testForm.get('newBookName')?.value;
    // if user leaves without typing, hide the input again
    if (!name) {
      this.showAddBookInput = false;
      this.showAddBookTestInput = false;
    }
  }

  onNewBookTestBlur() {
    const name = this.testForm.get('newBookTestName')?.value;
    // if user leaves without typing, hide the input again
    if (!name) {
      this.showAddBookTestInput = false;
    }
  }

  openNewBookTestAdd() {
    this.showAddBookTestInput = true;
    this.testForm.patchValue({ bookTestId: null });
  }

  private createTestPayload(): Test {
    return {
      id: this.id,
      name: this.testForm.value.name,
      description: this.testForm.value.description,
      gradeId: this.testForm.value.gradeId,
      maxDurationSeconds: +this.testForm.value.maxDurationMinutes * 60,
      isPracticeTest: this.testForm.value.isPracticeTest,
      imageUrl: this.testForm.value.imageUrl,
      subtitle: this.testForm.value.subtitle,
      newBookName: this.testForm.value.newBookName,
      newBookTestName: this.testForm.value.newBookTestName,
      bookTestId: !this.testForm.value.bookTestId ? 0 : this.testForm.value.bookTestId,
      bookId: !this.testForm.value.bookId ? 0 : this.testForm.value.bookId,
    };
  }

  navigateToQuestionCanvas() {
    if (this.testForm.valid) {
      this.testService.create(this.createTestPayload()).subscribe(() => {
        const navigationExtras: NavigationExtras = {
          state: {
            subjectId: null,
            topicId: null,
            subtopicId: null,
            testId: this.testForm.value.subtitle,
            bookId: this.testForm.value.bookId,
            bookTestId: this.testForm.value.bookTestId,
            testValue: this.id,
          },
        };
        setTimeout(() => {
          this.router.navigate(['/questioncanvas'], navigationExtras);
        }, 1000);
      });
    }
  }

  onSubmit() {
    if (this.testForm.valid) {
      this.testService.create(this.createTestPayload()).subscribe((response) => {
        if (this.isEditMode) {
          this.reload();
        } else {
          this.router.navigate(['/exam', response.examId]);
        }
      });
    }
  }

  navigateToTestList() {
    this.router.navigate(['/test-list']); // Yeni test oluşturma sayfasına yönlendir
  }

  onImageUploadForTest(event: any) {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        this.testForm.patchValue({ imageUrl: reader.result });
      };
      reader.readAsDataURL(file);
    }
  }

  onBulkFileSelected(event: any) {
    const file = event.target.files[0];
    if (file) {
      this.selectedBulkFile = file;
    }
  }

  async onBulkImport() {
    if (!this.selectedBulkFile) return;
    this.isUploading = true;
    this.bulkImportResults = null;
    try {
      const data = await this.readExcelFile(this.selectedBulkFile);
      // Excel'den gelen veriyi backend DTO formatına çevir
      const exams = data.map((row: any) => ({
        name: row['Ad'] || row['Ders'],
        description: row['Açıklama'] || row['Description'] || '',
        gradeId: +row['Sınıf'] || +row['Grade'] || 1,
        maxDurationSeconds: +row['Süre'] ? +row['Duration'] * 60 : 600,
        isPracticeTest: row['ÇalışmaTesti'] === 'Evet' || row['IsPracticeTest'] === true,
        subtitle: row['AltBaşlık'] || row['Subtitle'] || '',
        badgeText: row['Badge'] || '',
        bookTestId: row['BookTestId'] ? +row['BookTestId'] : undefined,
        bookId: row['BookId'] ? +row['BookId'] : undefined,
        newBookName: row['Book'] || '',
        newBookTestName: row['BookTest'] || '',
      }));
      const payload = { exams };
      this.testService.bulkImport(payload).subscribe({
        next: (res) => {
          this.bulkImportResults = res;
          this.isUploading = false;
        },
        error: (err) => {
          this.bulkImportResults = { isSuccess: false, message: 'Yükleme sırasında hata oluştu.' };
          this.isUploading = false;
        },
      });
    } catch (e) {
      this.bulkImportResults = { isSuccess: false, message: 'Excel dosyası okunamadı.' };
      this.isUploading = false;
    }
  }

  private readExcelFile(file: File): Promise<any[]> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e: any) => {
        const data = new Uint8Array(e.target.result);
        const workbook = XLSX.read(data, { type: 'array' });
        const sheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[sheetName];
        const json = XLSX.utils.sheet_to_json(worksheet, { defval: '' });
        resolve(json);
      };
      reader.onerror = (err) => reject(err);
      reader.readAsArrayBuffer(file);
    });
  }
}

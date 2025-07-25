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
import { GradesService } from '../../services/grades.service';
import { SubjectService } from '../../services/subject.service';
import { Subject } from '../../models/subject';
import { Topic } from '../../models/topic';
import { SubTopic } from '../../models/subtopic';
import { MatChipsModule } from '@angular/material/chips';
import { MatTableModule } from '@angular/material/table';
import { MatRadioModule } from '@angular/material/radio';
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
    MatChipsModule,
    MatTableModule,
    MatRadioModule,
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
  gradeService = inject(GradesService);
  subjectService = inject(SubjectService);
  grades = [] as any[];
  subjects = [] as Subject[];
  topics = [] as Topic[];
  subtopics = [] as SubTopic[];
  showAddBookInput = false;
  showAddBookTestInput = false;
  /** grab the input once it’s in the DOM */
  @ViewChild('newBookInput') newBookInput!: ElementRef<HTMLInputElement>;
  selectedBulkFile: File | null = null;
  isUploading = false;
  bulkImportResults: any = null;
  bulkImportData: any[] = [];
  selectedBulkIndex: number | null = null;
  lastPatchedBulkFormValue: any = null;

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
      subjectId: [''],
      topicId: [''],
      subtopicId: [''],
      questionCount: [0],
    });
  }

  ngOnInit(): void {
    this.id = this.route.snapshot.paramMap.get('id') ? Number(this.route.snapshot.paramMap.get('id')) : null;
    this.isEditMode = this.id !== null;
    this.reload();
    this.loadGrades();
    this.loadSubjects();

    // Form değişikliklerini dinle
    this.testForm.valueChanges.subscribe(() => {
      if (this.selectedBulkIndex !== null && this.bulkImportData[this.selectedBulkIndex]) {
        const current = this.testForm.value;
        const last = this.lastPatchedBulkFormValue;
        const changed = last && Object.keys(current).some((key) => current[key] !== last[key]);
        if (changed) {
          // this.bulkImportData field'ların name alanın tutarken form'da id değerleri var. gradeId, bookId, bookTestId gibi
          // bu yüzden form'da değişiklik yapıldığında bulkImportData'yı güncelle
          // Preserve name fields while updating with form values
          const updatedItem = {
            ...this.bulkImportData[this.selectedBulkIndex],
            name: current.name,
            description: current.description,
            maxDurationMinutes: current.maxDurationMinutes,
            isPracticeTest: current.isPracticeTest,
            subtitle: current.subtitle,
            imageUrl: current.imageUrl,
            __status: 'updated',
          };

          // Update ID fields while maintaining the corresponding name fields
          if (current.gradeId) {
            updatedItem.gradeId = current.gradeId;
            const grade = this.grades.find((g) => g.id === current.gradeId);
            if (grade) updatedItem.gradeName = grade.name;
          }

          if (current.bookId) {
            updatedItem.bookId = current.bookId;
            const book = this.books.find((b) => b.id === current.bookId);
            if (book) updatedItem.bookName = book.name;
          } else if (current.newBookName) {
            updatedItem.bookName = current.newBookName;
          }

          if (current.bookTestId) {
            updatedItem.bookTestId = current.bookTestId;
            const bookTest = this.bookTests.find((bt) => bt.id === current.bookTestId);
            if (bookTest) updatedItem.bookTestName = bookTest.name;
          } else if (current.newBookTestName) {
            updatedItem.bookTestName = current.newBookTestName;
          }

          this.bulkImportData[this.selectedBulkIndex] = updatedItem;
          this.bulkImportData = [...this.bulkImportData]; // tabloyu güncelle
        }
      }
    });
  }

  loadSubjects() {
    this.subjectService.loadCategories().subscribe((data) => {
      this.subjects = data;
      if (this.testForm.value.subjectId) {
        this.onSubjectChange(this.testForm.value.subjectId);
      }
    });
  }

  onSubjectChange(subjectId: number) {
    const gradeId = this.testForm.value.gradeId;
    if (subjectId && gradeId) {
      this.subjectService.getTopicsBySubjectAndGrade(subjectId, gradeId).subscribe((topics) => {
        this.topics = topics;
        this.testForm.patchValue({ topicId: null, subtopicId: null });
      });
    } else {
      this.topics = [];
      this.testForm.patchValue({ topicId: null, subtopicId: null });
    }
  }
  onTopicChange(topicId: number) {
    if (topicId) {
      this.subjectService.getSubTopicsByTopic(topicId).subscribe((subtopics) => {
        this.subtopics = subtopics;
        this.testForm.patchValue({ subtopicId: null });
      });
    } else {
      this.subtopics = [];
      this.testForm.patchValue({ subtopicId: null });
    }
  }
  onSubtopicChange(subtopicId: number) {
    if (subtopicId) {
      // Do any additional logic needed when subtopic changes
    } else {
      // Reset any dependent fields if needed
    }
  }

  reload() {
    this.showAddBookInput = false;
    this.showAddBookTestInput = false;
    this.loadBooks();
    if (this.isEditMode) {
      this.loadTest();
    }
  }
  loadGrades() {
    this.gradeService.getGrades().subscribe((data) => {
      this.grades = data;
      if (this.testForm.value.gradeId) {
        this.gradeId = this.testForm.value.gradeId;
      }
    });
  }

  get getSelectedGradeName(): string {
    const selectedGrade = this.grades.find((grade) => grade.id === this.testForm.value.gradeId);
    return selectedGrade ? selectedGrade.name : 'Sınıf Seçin';
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
        // Eğer kitap değiştirildiyse, kitap testi seçimini sıfırla
        this.testForm.patchValue({ bookTestId: null, newBookTestName: '' });
        // Eğer excel'den gelen bir satır seçiliyse, bulkImportData'da da güncelle
        if (this.selectedBulkIndex !== null && this.bulkImportData[this.selectedBulkIndex]) {
          this.bulkImportData[this.selectedBulkIndex].bookTestId = null;
          this.bulkImportData[this.selectedBulkIndex].bookTestName = '';
          this.bulkImportData = [...this.bulkImportData]; // tabloyu güncelle
        }
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
      subjectId: this.testForm.value.subjectId ? this.testForm.value.subjectId : null,
      topicId: this.testForm.value.topicId ? this.testForm.value.topicId : null,
      subtopicId: this.testForm.value.subtopicId ? this.testForm.value.subtopicId : null,
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

  // Excel'den gelen her satırın orijinal halini __original alanında sakla. Böylece değişiklikleri karşılaştırmak mümkün olacak.
  async onBulkImport() {
    if (!this.selectedBulkFile) return;
    this.isUploading = true;
    this.bulkImportResults = null;
    try {
      const data = await this.readExcelFile(this.selectedBulkFile);
      // Excel'den gelen veriyi backend DTO formatına çevir
      const exams = data.map((row: any) => {
        const item = {
          name: row['Ad'] || row['Ders'],
          description: row['Açıklama'] || row['Description'] || '',
          gradeId: row['Sınıf'] || row['Grade'] || '',
          maxDurationMinutes: +row['Süre'] || +row['Duration'] || 10,
          isPracticeTest: row['ÇalışmaTesti'] === 'Evet' || row['IsPracticeTest'] === true,
          subtitle: row['AltBaşlık'] || row['Subtitle'] || '',
          badgeText: row['Badge'] || '',
          bookTestId: row['BookTestId'] ? +row['BookTestId'] : undefined,
          bookId: row['BookId'] ? +row['BookId'] : undefined,
          bookName: row['Book'] || '',
          bookTestName: row['BookTest'] || '',
          newBookName: row['YeniKitap'] || '',
          newBookTestName: row['YeniKitapTesti'] || '',
        };
        return {
          ...item,
          __original: { ...item },
        };
      });
      this.bulkImportData = exams;
      this.ensureBulkImportNames();
      this.isUploading = false;
    } catch (e) {
      this.bulkImportResults = { success: false, message: 'Excel dosyası okunamadı.' };
      this.isUploading = false;
    }
  }

  onBulkItemSelect(i: number, element: any) {
    this.selectedBulkIndex = i;

    let bookId = 0;
    let bookTestId = 0;
    let newBookName = '';
    let newBookTestName = '';
    let gradeId = 0;
    let showAddBookInput = false;
    let showAddBookTestInput = false;

    // Kitap adı ile eşleşen kitap var mı?
    if (element.bookName) {
      const normalize = (str: string) =>
        str
          ?.toLocaleLowerCase('tr-TR')
          .replace(/ı/g, 'i')
          .replace(/İ/g, 'i')
          .replace(/ş/g, 's')
          .replace(/Ş/g, 's')
          .replace(/ğ/g, 'g')
          .replace(/Ğ/g, 'g')
          .replace(/ü/g, 'u')
          .replace(/Ü/g, 'u')
          .replace(/ö/g, 'o')
          .replace(/Ö/g, 'o')
          .replace(/ç/g, 'c')
          .replace(/Ç/g, 'c')
          .trim();
      const foundBook = this.books.find((b) => normalize(b.name) === normalize(element.bookName));
      if (foundBook) {
        bookId = foundBook.id;
        newBookName = '';
        showAddBookInput = false;
        // Kitap testleri de yüklenmeli
        this.bookService.getTestsByBook(foundBook.id).subscribe((tests) => {
          this.bookTests = tests;
          // Kitap testi adı ile eşleşen test var mı? (bookId varsa ona göre, yoksa genel listede arar)
          if (element.bookTestName) {
            let foundBookTest;
            const normalize = (str: string) =>
              str
                ?.toLocaleLowerCase('tr-TR')
                .replace(/ı/g, 'i')
                .replace(/İ/g, 'i')
                .replace(/ş/g, 's')
                .replace(/Ş/g, 's')
                .replace(/ğ/g, 'g')
                .replace(/Ğ/g, 'g')
                .replace(/ü/g, 'u')
                .replace(/Ü/g, 'u')
                .replace(/ö/g, 'o')
                .replace(/Ö/g, 'o')
                .replace(/ç/g, 'c')
                .replace(/Ç/g, 'c')
                .trim();
            if (bookId) {
              foundBookTest = this.bookTests.find((bt) => normalize(bt.name) === normalize(element.bookTestName));
            } else {
              foundBookTest = null;
            }
            if (foundBookTest) {
              bookTestId = foundBookTest.id;
              newBookTestName = '';
              showAddBookTestInput = false;
              this.testForm.patchValue({
                bookId: bookId,
                bookTestId: bookTestId,
              });
            } else {
              bookTestId = 0;
              newBookTestName = element.bookTestName;
              showAddBookTestInput = true;
            }
          }
        });
      } else {
        bookId = 0;
        newBookName = element.bookName;
        showAddBookInput = true;
        this.bookTests = [];
        // Kitap bulunamazsa kitap testi de yeni olarak işaretlenir
        newBookTestName = element.bookTestName;
        showAddBookTestInput = true;
        this.testForm.patchValue({
          bookId: null,
          bookTestId: null,
          newBookName: newBookName,
          newBookTestName: newBookTestName,
        });
        this.showAddBookInput = showAddBookInput;
        this.showAddBookTestInput = showAddBookTestInput;
      }
    }

    // bookId ve bookTestId gibi grade id'yi de kontrol et excelde grade'in name alanı geliyor ama form'da id gerekiyor listeden bulsun ve setlesin aynı şekilde
    // case insensitive olarak kontrol et ve türkçe karakterleri normalize et
    if (element.gradeId && typeof element.gradeId === 'string') {
      element.gradeId = element.gradeId.trim();
    }
    if (element.grade && typeof element.grade === 'string') {
      element.grade = element.grade.trim();
    }
    if (element.gradeId) {
      const normalize = (str: string) =>
        str
          ?.toLocaleLowerCase('tr-TR')
          .replace(/ı/g, 'i')
          .replace(/İ/g, 'i')
          .replace(/ş/g, 's')
          .replace(/Ş/g, 's')
          .replace(/ğ/g, 'g')
          .replace(/Ğ/g, 'g')
          .replace(/ü/g, 'u')
          .replace(/Ü/g, 'u')
          .replace(/ö/g, 'o')
          .replace(/Ö/g, 'o')
          .replace(/ç/g, 'c')
          .replace(/Ç/g, 'c')
          .trim();
      const foundGrade = this.grades.find((g) => normalize(g.name) === normalize(element.gradeId));
      if (foundGrade) {
        gradeId = foundGrade.id;
        this.testForm.patchValue({ gradeId: foundGrade.id });
      } else {
        gradeId = 0; // Eğer bulunamazsa varsayılan olarak 0
        this.testForm.patchValue({ gradeId: null });
      }
    }

    // Formu patchle
    this.testForm.patchValue({
      name: element.name,
      description: element.description,
      gradeId: gradeId,
      maxDurationMinutes: element.maxDurationMinutes,
      isPracticeTest: element.isPracticeTest,
      subtitle: element.subtitle,
      bookTestId: bookTestId,
      bookId: bookId,
      newBookName: newBookName,
      newBookTestName: newBookTestName,
      imageUrl: element.imageUrl,
      subjectId: element.subjectId ? +element.subjectId : null,
      topicId: element.topicId ? +element.topicId : null,
      subtopicId: element.subtopicId ? +element.subtopicId : null,
    });
    this.showAddBookInput = showAddBookInput;
    this.showAddBookTestInput = showAddBookTestInput;
    // Seçildiğinde statü değiştirme! Sadece son patchlenen değeri sakla
    this.lastPatchedBulkFormValue = { ...this.testForm.value };
  }

  processBulkImport() {
    if (this.selectedBulkIndex !== null && this.bulkImportData[this.selectedBulkIndex]) {
      this.bulkImportData[this.selectedBulkIndex] = {
        ...this.bulkImportData[this.selectedBulkIndex],
        ...this.testForm.value,
        __status: 'updated',
      };
      this.bulkImportData = [...this.bulkImportData]; // tabloyu güncelle
      this.ensureBulkImportNames();
      this.lastPatchedBulkFormValue = { ...this.testForm.value };
    }
  }

  hasUpdatedItems() {
    return this.bulkImportData.some((item) => item.__status === 'updated');
  }

  getBulkItemStatus(i: number) {
    if (!this.bulkImportData[i]) return 'pending';
    if (this.selectedBulkIndex === i) {
      // Formda değişiklik var mı kontrolü
      if (this.lastPatchedBulkFormValue) {
        const current = this.testForm.value;
        const last = this.lastPatchedBulkFormValue;
        const changed = Object.keys(current).some((key) => current[key] !== last[key]);
        if (changed) return 'updated';
      }
    }
    return this.bulkImportData[i].__status === 'updated' ? 'updated' : 'pending';
  }

  // Yardımcı fonksiyon: Bir alan değişti mi?
  isFieldChanged(field: string, i: number): boolean {
    const item = this.bulkImportData[i];
    if (!item || !item.__original) return false;
    return item[field] !== item.__original[field];
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

  clearBulkImport() {
    this.bulkImportData = [];
    this.selectedBulkIndex = null;
  }

  // Excel'den gelen veya dışarıdan eklenen her item'a gradeName, bookName, bookTestName ekle
  private ensureBulkImportNames() {
    if (!this.bulkImportData || !this.grades || !this.books) return;
    this.bulkImportData = this.bulkImportData.map((item) => {
      // GRADE
      let gradeName = item.gradeName || '';
      if (!gradeName && item.gradeId) {
        const found = this.grades.find((g) => g.id === item.gradeId || g.name === item.gradeId);
        if (found) gradeName = found.name;
      }
      if (!gradeName && item.grade) gradeName = item.grade;

      // BOOK
      let bookName = item.bookName || '';
      if (!bookName && item.bookId) {
        const foundBook = this.books.find((b) => b.id === item.bookId || b.name === item.bookId);
        if (foundBook) bookName = foundBook.name;
      }
      if (!bookName && item.newBookName) bookName = item.newBookName;

      // BOOK TEST
      let bookTestName = item.bookTestName || '';
      if (!bookTestName && item.bookTestId && this.bookTests) {
        const foundBookTest = this.bookTests.find((bt) => bt.id === item.bookTestId || bt.name === item.bookTestId);
        if (foundBookTest) bookTestName = foundBookTest.name;
      }
      if (!bookTestName && item.newBookTestName) bookTestName = item.newBookTestName;

      return { ...item, gradeName, bookName, bookTestName };
    });
  }
}
